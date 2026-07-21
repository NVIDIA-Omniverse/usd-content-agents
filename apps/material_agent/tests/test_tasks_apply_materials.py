# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ApplyMaterialsToUSD task error handling and stage metadata."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

import material_agent.tasks.apply_materials_to_usd as apply_module
from material_agent.materials import FALLBACK_MATERIAL_NAME
from material_agent.tasks.apply_materials_to_usd import ApplyMaterialsToUSDTask


class TestApplyMaterialsErrorHandling:
    """Tests for error handling in ApplyMaterialsToUSDTask."""

    def test_fails_when_predictions_exist_but_no_materials_resolved(self, tmp_path):
        """Test that task fails with clear error when predictions exist but materials cannot be resolved.

        This tests the fix for Issue #2 where the pipeline should fail when:
        - VLM successfully generates predictions
        - But material resolution fails (materials don't match library)
        - Previously it would silently continue with no materials
        """
        # Setup: Create predictions file with valid predictions
        predictions_path = tmp_path / "predictions.jsonl"
        predictions = [
            {
                "id": "/RootNode/Geometry/Part1",
                "materials": {
                    "material": "NonExistentMaterial",
                    "original_response": "Some reasoning",
                },
            },
            {
                "id": "/RootNode/Geometry/Part2",
                "materials": {
                    "material": "AnotherMissingMaterial",
                    "original_response": "More reasoning",
                },
            },
        ]

        with open(predictions_path, "w") as f:
            for pred in predictions:
                f.write(json.dumps(pred) + "\n")

        # Setup: Create mock USD files
        input_usd = tmp_path / "input.usd"
        input_usd.write_text("# Mock USD")
        output_usd = tmp_path / "output.usd"

        # Setup: Context with predictions but NO resolved materials (resolution failed)
        context = {
            "input_usd_path": str(input_usd),
            "output_usd_path": str(output_usd),
            "predictions_path": str(predictions_path),
            "resolved_materials": {},  # Empty - material resolution failed!
            "is_library_based_mapping": True,
            "material_library_path": "/path/to/library.usd",
        }

        # Create task
        task = ApplyMaterialsToUSDTask()

        # Execute and verify it raises ValueError with clear error message
        with pytest.raises(ValueError) as exc_info:
            task.run(context)

        # Verify error message contains key information
        error_msg = str(exc_info.value)
        assert "Critical error" in error_msg
        assert "Material resolution failed" in error_msg
        assert "VLM predicted materials but none could be resolved" in error_msg
        assert "check system prompt" in error_msg.lower()
        assert "MaterialRetrieval task logs" in error_msg

    def test_all_unknown_predictions_use_fallback(self, tmp_path: Path):
        predictions_path = tmp_path / "predictions.jsonl"
        predictions_path.write_text(
            json.dumps(
                {
                    "id": "/RootNode/Geometry/Part1",
                    "materials": {
                        "material": "__UNKNOWN__",
                        "reason": "no visible geometry",
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        input_usd = tmp_path / "input.usd"
        input_usd.write_text("#usda 1.0\n")

        context = {
            "input_usd_path": str(input_usd),
            "output_usd_path": str(tmp_path / "output.usd"),
            "predictions_path": str(predictions_path),
            "resolved_materials": {},
            "is_library_based_mapping": True,
        }

        result = ApplyMaterialsToUSDTask().run(context)

        assert FALLBACK_MATERIAL_NAME in result["materials_applied"]
        assert result["assignment_stats"]["unknown"] == 1

    def test_empty_apply_allowed_still_uses_fallback_for_unknown_predictions(
        self, tmp_path: Path
    ) -> None:
        predictions_path = tmp_path / "predictions.jsonl"
        predictions_path.write_text(
            json.dumps({"id": "/Hidden", "materials": {"material": "__UNKNOWN__"}})
            + "\n",
            encoding="utf-8",
        )
        input_usd = tmp_path / "input.usd"
        input_usd.write_text("#usda 1.0\n")

        context = {
            "input_usd_path": str(input_usd),
            "output_usd_path": str(tmp_path / "output.usd"),
            "predictions_path": str(predictions_path),
            "resolved_materials": {},
            "is_library_based_mapping": True,
            "allow_empty_predictions": True,
        }

        result = ApplyMaterialsToUSDTask().run(context)

        assert FALLBACK_MATERIAL_NAME in result["materials_applied"]
        assert result["assignment_stats"]["unknown"] == 1
        assert result["assignment_stats"]["materials_applied"] >= 0

    def test_load_mapping_uses_fallback_for_unknown_predictions(
        self, tmp_path: Path
    ) -> None:
        predictions_path = tmp_path / "predictions.jsonl"
        predictions_path.write_text(
            json.dumps(
                {
                    "id": "/RootNode/Geometry/Hidden",
                    "materials": {"material": "__UNKNOWN__"},
                }
            )
            + "\n"
            + json.dumps(
                {
                    "id": "/RootNode/Geometry/Visible",
                    "materials": {"material": "Steel"},
                }
            )
            + "\n",
            encoding="utf-8",
        )

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        mapping = task._load_prim_material_mapping(predictions_path)

        assert mapping == {
            "/RootNode/Geometry/Hidden": FALLBACK_MATERIAL_NAME,
            "/RootNode/Geometry/Visible": "Steel",
        }

    def test_load_mapping_skips_unresolved_default_fallback(
        self, tmp_path: Path
    ) -> None:
        predictions_path = tmp_path / "predictions.jsonl"
        predictions_path.write_text(
            json.dumps(
                {
                    "id": "/RootNode/Geometry/Fallback",
                    "materials": {"material": "__USE_DEFAULT_LIBRARY__"},
                }
            )
            + "\n"
            + json.dumps(
                {
                    "id": "/RootNode/Geometry/Visible",
                    "materials": {"material": "Steel"},
                }
            )
            + "\n",
            encoding="utf-8",
        )

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        mapping = task._load_prim_material_mapping(predictions_path)
        counts = task._count_prediction_materials(predictions_path)

        assert mapping == {"/RootNode/Geometry/Visible": "Steel"}
        assert counts["unknown"] == 1
        assert counts["actionable"] == 1
        assert counts["missing"] == 0

    def test_load_mapping_normalizes_material_names(self, tmp_path: Path) -> None:
        predictions_path = tmp_path / "predictions.jsonl"
        predictions_path.write_text(
            json.dumps(
                {
                    "id": "/RootNode/Geometry/Visible",
                    "materials": {"material": " Steel "},
                }
            )
            + "\n",
            encoding="utf-8",
        )

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        mapping = task._load_prim_material_mapping(predictions_path)

        assert mapping == {"/RootNode/Geometry/Visible": "Steel"}

    def test_load_mapping_handles_predicted_material_field(
        self, tmp_path: Path
    ) -> None:
        predictions_path = tmp_path / "predictions.jsonl"
        predictions_path.write_text(
            json.dumps(
                {
                    "id": "/RootNode/Geometry/Visible",
                    "predicted_material": " Steel ",
                }
            )
            + "\n",
            encoding="utf-8",
        )

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        mapping = task._load_prim_material_mapping(predictions_path)
        counts = task._count_prediction_materials(predictions_path)

        assert mapping == {"/RootNode/Geometry/Visible": "Steel"}
        assert counts["total"] == 1
        assert counts["actionable"] == 1
        assert counts["missing"] == 0

    def test_load_mapping_handles_nested_payloads(self, tmp_path: Path) -> None:
        predictions_path = tmp_path / "predictions.jsonl"
        predictions_path.write_text(
            json.dumps(
                {
                    "predictions": [
                        {
                            "id": "/RootNode/Geometry/Hidden",
                            "materials": {"material": "__UNKNOWN__"},
                        },
                        {
                            "id": "/RootNode/Geometry/Visible",
                            "materials": {"material": " Steel "},
                        },
                    ]
                }
            )
            + "\n"
            + json.dumps(
                {
                    "/MappedHidden": {
                        "materials": {"material": "__UNKNOWN__"},
                    },
                    "/MappedVisible": "Plastic",
                }
            )
            + "\n",
            encoding="utf-8",
        )

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        mapping = task._load_prim_material_mapping(predictions_path)

        assert mapping == {
            "/RootNode/Geometry/Hidden": FALLBACK_MATERIAL_NAME,
            "/RootNode/Geometry/Visible": "Steel",
            "/MappedHidden": FALLBACK_MATERIAL_NAME,
            "/MappedVisible": "Plastic",
        }

    def test_load_mapping_propagates_parent_id_into_container_children(
        self, tmp_path: Path
    ) -> None:
        predictions_path = tmp_path / "predictions.jsonl"
        predictions_path.write_text(
            json.dumps(
                {
                    "id": "/RootNode/Geometry/Parent",
                    "predictions": [
                        {
                            "materials": {"material": " Steel "},
                        }
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        mapping = task._load_prim_material_mapping(predictions_path)
        counts = task._count_prediction_materials(predictions_path)

        assert mapping == {"/RootNode/Geometry/Parent": "Steel"}
        assert counts["total"] == 1
        assert counts["actionable"] == 1

    def test_count_prediction_materials_classifies_jsonl_rows(
        self, tmp_path: Path
    ) -> None:
        predictions_path = tmp_path / "predictions.jsonl"
        predictions_path.write_text(
            json.dumps({"id": "/Hidden", "materials": {"material": "__UNKNOWN__"}})
            + "\n"
            + json.dumps({"id": "/Visible", "materials": {"material": "Steel"}})
            + "\n"
            + json.dumps({"id": "/NoMaterial", "materials": {}})
            + "\n"
            + "{invalid json}\n",
            encoding="utf-8",
        )

        counts = ApplyMaterialsToUSDTask()._count_prediction_materials(predictions_path)

        assert counts == {"total": 3, "actionable": 1, "unknown": 1, "missing": 1}

    def test_count_prediction_materials_classifies_nested_payloads(
        self, tmp_path: Path
    ) -> None:
        predictions_path = tmp_path / "predictions.jsonl"
        predictions_path.write_text(
            json.dumps(
                {
                    "predictions": [
                        {
                            "id": "/Hidden",
                            "materials": {"material": "__UNKNOWN__"},
                        },
                        {"id": "/Visible", "material": "Steel"},
                        {"id": "/NoMaterial", "materials": {}},
                    ]
                }
            )
            + "\n"
            + json.dumps(
                {
                    "/MappedHidden": {
                        "materials": {"material": "__UNKNOWN__"},
                    },
                    "/MappedVisible": "Plastic",
                }
            )
            + "\n",
            encoding="utf-8",
        )

        counts = ApplyMaterialsToUSDTask()._count_prediction_materials(predictions_path)

        assert counts == {"total": 5, "actionable": 2, "unknown": 2, "missing": 1}

    def test_path_keyed_peers_are_counted_when_top_level_material_exists(
        self, tmp_path: Path
    ) -> None:
        predictions_path = tmp_path / "predictions.jsonl"
        predictions_path.write_text(
            json.dumps(
                {
                    "id": "/Batch",
                    "material": "Batch Material",
                    "/MappedHidden": {
                        "material": "",
                        "validation_status": "disallowed_unknown",
                    },
                    "/MappedVisible": "Steel",
                }
            )
            + "\n",
            encoding="utf-8",
        )

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        mapping = task._load_prim_material_mapping(predictions_path)
        counts = task._count_prediction_materials(predictions_path)

        assert mapping == {
            "/Batch": "Batch Material",
            "/MappedHidden": FALLBACK_MATERIAL_NAME,
            "/MappedVisible": "Steel",
        }
        assert counts == {"total": 3, "actionable": 2, "unknown": 1, "missing": 0}

    def test_count_prediction_materials_ignores_metadata_only_path_peers(
        self, tmp_path: Path
    ) -> None:
        predictions_path = tmp_path / "predictions.jsonl"
        predictions_path.write_text(
            json.dumps(
                {
                    "id": "/Batch",
                    "material": "Batch Material",
                    "/MetadataOnly": {"validation_status": "valid"},
                }
            )
            + "\n",
            encoding="utf-8",
        )

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        mapping = task._load_prim_material_mapping(predictions_path)
        counts = task._count_prediction_materials(predictions_path)

        assert mapping == {"/Batch": "Batch Material"}
        assert counts == {"total": 1, "actionable": 1, "unknown": 0, "missing": 0}

    def test_count_prediction_materials_ignores_bare_empty_dicts(
        self, tmp_path: Path
    ) -> None:
        predictions_path = tmp_path / "predictions.jsonl"
        predictions_path.write_text(
            json.dumps(
                [
                    {},
                    {"id": "/NoMaterial"},
                    {"materials": {}},
                    "Steel",
                    "__UNKNOWN__",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        counts = ApplyMaterialsToUSDTask()._count_prediction_materials(predictions_path)

        assert counts == {"total": 4, "actionable": 1, "unknown": 1, "missing": 2}

    def test_load_mapping_handles_string_items_in_parent_container(
        self, tmp_path: Path
    ) -> None:
        predictions_path = tmp_path / "predictions.jsonl"
        predictions_path.write_text(
            json.dumps({"id": "/Parent", "predictions": ["Steel", "__UNKNOWN__"]})
            + "\n",
            encoding="utf-8",
        )

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        mapping = task._load_prim_material_mapping(predictions_path)
        counts = task._count_prediction_materials(predictions_path)

        assert mapping == {"/Parent": FALLBACK_MATERIAL_NAME}
        assert counts == {"total": 2, "actionable": 1, "unknown": 1, "missing": 0}

    def test_count_prediction_materials_warns_on_invalid_json(
        self, tmp_path: Path
    ) -> None:
        predictions_path = tmp_path / "predictions.jsonl"
        predictions_path.write_text("{invalid json}\n", encoding="utf-8")
        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        counts = task._count_prediction_materials(predictions_path)

        assert counts == {"total": 0, "actionable": 0, "unknown": 0, "missing": 0}
        task.listener.warning.assert_called_once()
        assert (
            task.listener.warning.call_args[0][0]
            == "Failed to parse prediction line while counting materials: "
            "Expecting property name enclosed in double quotes: line 1 column 2 "
            "(char 1)"
        )

    def test_strict_unknown_mode_fails_on_partial_unknown_predictions(
        self, tmp_path: Path
    ) -> None:
        predictions_path = tmp_path / "predictions.jsonl"
        predictions_path.write_text(
            json.dumps({"id": "/Hidden", "materials": {"material": "__UNKNOWN__"}})
            + "\n"
            + json.dumps({"id": "/Visible", "materials": {"material": "Steel"}})
            + "\n",
            encoding="utf-8",
        )
        input_usd = tmp_path / "input.usd"
        input_usd.write_text("#usda 1.0\n")

        context = {
            "input_usd_path": str(input_usd),
            "output_usd_path": str(tmp_path / "output.usd"),
            "predictions_path": str(predictions_path),
            "resolved_materials": {"Steel": "/path/to/steel.usd"},
            "is_library_based_mapping": True,
            "fail_on_unknown_material": True,
        }

        with pytest.raises(ValueError, match="fail_on_unknown_material=true"):
            ApplyMaterialsToUSDTask().run(context)

    def test_strict_unknown_mode_fails_on_nested_unknown_predictions(
        self, tmp_path: Path
    ) -> None:
        predictions_path = tmp_path / "predictions.jsonl"
        predictions_path.write_text(
            json.dumps(
                {
                    "predictions": [
                        {
                            "id": "/Hidden",
                            "materials": {"material": "__UNKNOWN__"},
                        },
                        {"id": "/Visible", "materials": {"material": "Steel"}},
                    ]
                }
            )
            + "\n",
            encoding="utf-8",
        )
        input_usd = tmp_path / "input.usd"
        input_usd.write_text("#usda 1.0\n")

        context = {
            "input_usd_path": str(input_usd),
            "output_usd_path": str(tmp_path / "output.usd"),
            "predictions_path": str(predictions_path),
            "resolved_materials": {"Steel": "/path/to/steel.usd"},
            "is_library_based_mapping": True,
            "fail_on_unknown_material": True,
        }

        with pytest.raises(ValueError, match="fail_on_unknown_material=true"):
            ApplyMaterialsToUSDTask().run(context)

    def test_strict_unknown_mode_fails_when_validation_cleared_sentinel(
        self, tmp_path: Path
    ) -> None:
        predictions_path = tmp_path / "predictions.jsonl"
        predictions_path.write_text(
            json.dumps({"id": "/Hidden", "materials": {"material": ""}}) + "\n",
            encoding="utf-8",
        )
        input_usd = tmp_path / "input.usd"
        input_usd.write_text("#usda 1.0\n")

        context = {
            "input_usd_path": str(input_usd),
            "output_usd_path": str(tmp_path / "output.usd"),
            "predictions_path": str(predictions_path),
            "resolved_materials": {"Steel": "/path/to/steel.usd"},
            "is_library_based_mapping": True,
            "fail_on_unknown_material": True,
            "unknown_material_predictions": 1,
        }

        with pytest.raises(ValueError) as exc_info:
            ApplyMaterialsToUSDTask().run(context)

        error_msg = str(exc_info.value)
        assert "fail_on_unknown_material=true" in error_msg
        assert "earlier validation steps" in error_msg

    def test_strict_unknown_mode_fails_on_durable_disallowed_unknown_marker(
        self, tmp_path: Path
    ) -> None:
        predictions_path = tmp_path / "predictions.jsonl"
        predictions_path.write_text(
            json.dumps(
                {
                    "id": "/Hidden",
                    "materials": {
                        "material": "",
                        "validation_status": "disallowed_unknown",
                    },
                }
            )
            + "\n"
            + json.dumps({"id": "/Visible", "materials": {"material": "Steel"}})
            + "\n",
            encoding="utf-8",
        )
        input_usd = tmp_path / "input.usd"
        input_usd.write_text("#usda 1.0\n")

        context = {
            "input_usd_path": str(input_usd),
            "output_usd_path": str(tmp_path / "output.usd"),
            "predictions_path": str(predictions_path),
            "resolved_materials": {"Steel": "/path/to/steel.usd"},
            "is_library_based_mapping": True,
            "fail_on_unknown_material": True,
        }

        task = ApplyMaterialsToUSDTask()
        counts = task._count_prediction_materials(predictions_path)

        assert counts == {"total": 2, "actionable": 1, "unknown": 1, "missing": 0}
        with pytest.raises(ValueError, match="fail_on_unknown_material=true"):
            task.run(context)

    def test_fails_clearly_when_predictions_have_only_missing_materials(
        self, tmp_path: Path
    ) -> None:
        predictions_path = tmp_path / "predictions.jsonl"
        predictions_path.write_text(
            json.dumps({"id": "/Hidden", "materials": {"material": ""}}) + "\n",
            encoding="utf-8",
        )
        input_usd = tmp_path / "input.usd"
        input_usd.write_text("#usda 1.0\n")

        context = {
            "input_usd_path": str(input_usd),
            "output_usd_path": str(tmp_path / "output.usd"),
            "predictions_path": str(predictions_path),
            "resolved_materials": {},
            "is_library_based_mapping": True,
        }

        with pytest.raises(
            ValueError,
            match="did not contain actionable material values",
        ):
            ApplyMaterialsToUSDTask().run(context)

    def test_resolved_materials_with_unknown_predictions_adds_fallback(
        self, tmp_path: Path
    ) -> None:
        predictions_path = tmp_path / "predictions.jsonl"
        predictions_path.write_text(
            json.dumps({"id": "/Hidden", "materials": {"material": "__UNKNOWN__"}})
            + "\n",
            encoding="utf-8",
        )
        input_usd = tmp_path / "input.usd"
        input_usd.write_text("#usda 1.0\n")

        context = {
            "input_usd_path": str(input_usd),
            "output_usd_path": str(tmp_path / "output.usd"),
            "predictions_path": str(predictions_path),
            "resolved_materials": {"Steel": "/path/to/steel.usd"},
            "is_library_based_mapping": True,
        }

        result = ApplyMaterialsToUSDTask().run(context)

        assert FALLBACK_MATERIAL_NAME in result["materials_applied"]
        assert result["assignment_stats"]["unknown"] == 1

    def test_fails_when_no_predictions_and_no_materials_by_default(
        self, tmp_path: Path
    ) -> None:
        """No predictions should fail closed unless explicitly opted in."""
        # Setup: No predictions file
        predictions_path = tmp_path / "predictions.jsonl"
        # Don't create the file

        # Setup: Create mock USD files
        input_usd = tmp_path / "input.usd"
        input_usd.write_text("# Mock USD")
        output_usd = tmp_path / "output.usd"

        # Setup: Context with no resolved materials AND no predictions file
        context = {
            "input_usd_path": str(input_usd),
            "output_usd_path": str(output_usd),
            "predictions_path": str(
                predictions_path
            ),  # Path set but file doesn't exist
            "resolved_materials": {},  # Empty
            "is_library_based_mapping": True,
        }

        # Create task
        task = ApplyMaterialsToUSDTask()

        with pytest.raises(ValueError, match="No material predictions"):
            task.run(context)

    def test_allows_no_predictions_and_no_materials_when_opted_in(
        self, tmp_path: Path
    ) -> None:
        """Intentional empty material application remains an explicit opt-in."""
        predictions_path = tmp_path / "predictions.jsonl"
        input_usd = tmp_path / "input.usd"
        input_usd.write_text("# Mock USD")
        output_usd = tmp_path / "output.usd"
        context = {
            "input_usd_path": str(input_usd),
            "output_usd_path": str(output_usd),
            "predictions_path": str(predictions_path),
            "resolved_materials": {},
            "is_library_based_mapping": True,
            "allow_empty_predictions": True,
        }

        result = ApplyMaterialsToUSDTask().run(context)

        assert result is not None
        assert result["materials_applied"] == {}
        assert result["assignment_stats"]["total_prims"] == 0
        assert result["assignment_stats"]["materials_applied"] == 0
        assert result["assignment_stats"]["failed"] == 0

    def test_fails_when_resolved_materials_have_empty_predictions(
        self, tmp_path: Path
    ) -> None:
        """Resolved materials without prediction bindings should fail closed."""
        predictions_path = tmp_path / "predictions.jsonl"
        predictions_path.write_text("", encoding="utf-8")
        input_usd = tmp_path / "input.usd"
        input_usd.write_text("#usda 1.0\n")

        context = {
            "input_usd_path": str(input_usd),
            "output_usd_path": str(tmp_path / "output.usd"),
            "predictions_path": str(predictions_path),
            "resolved_materials": {"Steel": "/path/to/steel.usd"},
            "is_library_based_mapping": True,
        }

        with pytest.raises(ValueError, match="No material predictions"):
            ApplyMaterialsToUSDTask().run(context)

    def test_succeeds_when_predictions_and_materials_both_exist(self, tmp_path):
        """Test normal success case when predictions exist and materials are resolved."""
        # Setup: Create predictions file
        predictions_path = tmp_path / "predictions.jsonl"
        predictions = [
            {
                "id": "/RootNode/Geometry/Part1",
                "materials": {
                    "material": "Steel",
                    "original_response": "Some reasoning",
                },
            }
        ]

        with open(predictions_path, "w") as f:
            for pred in predictions:
                f.write(json.dumps(pred) + "\n")

        # Setup: Create mock USD files
        input_usd = tmp_path / "input.usd"
        output_usd = tmp_path / "output.usd"

        # Create minimal valid USD content
        usd_content = """#usda 1.0
(
    defaultPrim = "RootNode"
)

def Xform "RootNode" {
    def Xform "Geometry" {
        def Mesh "Part1" {
        }
    }
}
"""
        input_usd.write_text(usd_content)

        # Setup: Context with both predictions AND resolved materials
        context = {
            "input_usd_path": str(input_usd),
            "output_usd_path": str(output_usd),
            "predictions_path": str(predictions_path),
            "resolved_materials": {
                "Steel": "/path/to/steel.usd"  # Material successfully resolved!
            },
            "is_library_based_mapping": True,
            "material_library_path": str(tmp_path / "library.usd"),
            "layer_only": False,
            "flatten_output": False,
        }

        # Create task
        task = ApplyMaterialsToUSDTask()

        # Execute - should succeed without raising exceptions
        result = task.run(context)

        # Verify success (no exception raised means success)
        assert result is not None
        assert "materials_applied" in result
        assert "assignment_stats" in result
        # Note: Material binding might fail due to mock library path, but task should complete
        assert result["assignment_stats"]["total_prims"] >= 0

    def test_raises_when_usd_export_fails(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Authoring failures should abort instead of reporting success."""
        predictions_path = tmp_path / "predictions.jsonl"
        predictions_path.write_text(
            json.dumps(
                {
                    "id": "/RootNode/Geometry/Part1",
                    "materials": {"material": "Steel"},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        input_usd = tmp_path / "input.usd"
        output_usd = tmp_path / "output.usd"
        input_usd.write_text("#usda 1.0\n")

        def fail_export(path: str) -> None:
            Path(path).write_text("#usda 1.0\n# partial output\n", encoding="utf-8")
            raise RuntimeError("export failed")

        root_layer = MagicMock()
        root_layer.Export.side_effect = fail_export
        stage = MagicMock()
        stage.GetRootLayer.return_value = root_layer

        def fake_create_full_stage(*args, **kwargs):
            return (
                stage,
                {"/RootNode/Geometry/Part1": "Steel"},
                {"materials_created": 1, "prims_with_materials": 1, "failed": 0},
            )

        monkeypatch.setattr(
            ApplyMaterialsToUSDTask,
            "_create_full_stage",
            fake_create_full_stage,
        )

        with pytest.raises(RuntimeError, match="export failed"):
            ApplyMaterialsToUSDTask().run(
                {
                    "input_usd_path": str(input_usd),
                    "output_usd_path": str(output_usd),
                    "predictions_path": str(predictions_path),
                    "resolved_materials": {"Steel": "/path/to/steel.usd"},
                    "is_library_based_mapping": True,
                    "layer_only": False,
                    "flatten_output": False,
                }
            )
        assert not output_usd.exists()


class TestApplyMaterialsDatasetSystemPrompt:
    """Tests for system prompt loading from dataset.json."""

    def test_dataset_loading_extracts_system_prompt(self, tmp_path):
        """Test that DatasetLoadingTask extracts system prompt from dataset.json.

        This tests the fix for Issue #1 where system prompt from dataset.json
        should be loaded and passed to VLM inference.
        """
        from material_agent.tasks.dataset import DatasetLoadingTask

        # Setup: Create dataset.jsonl
        dataset_jsonl = tmp_path / "dataset.jsonl"
        dataset_entry = {
            "id": "test_entry",
            "media": {
                "images": [{"path": "image1.png", "metadata": {"view": "front"}}]
            },
            "user_prompt": "Identify the material",
        }

        with open(dataset_jsonl, "w") as f:
            f.write(json.dumps(dataset_entry) + "\n")

        # Setup: Create dataset.json with system prompt
        dataset_json = tmp_path / "dataset.json"
        dataset_metadata = {
            "schema_version": "0.2",
            "metadata": {"created": "2025-01-01", "num_entries": 1},
            "inference": {
                "prompts": [
                    {
                        "step_name": "material_selection",
                        "step_index": 0,
                        "system_prompt": "You are an expert at identifying materials. Return JSON format.",
                        "output_format": {"material": "material name"},
                    }
                ]
            },
            "prims_file": "dataset.jsonl",
        }

        with open(dataset_json, "w") as f:
            json.dump(dataset_metadata, f)

        # Create test images
        image1 = tmp_path / "image1.png"
        from PIL import Image

        Image.new("RGB", (100, 100), color="red").save(image1)

        # Setup: Context
        context = {"dataset_path": str(dataset_jsonl)}

        # Create and run task
        task = DatasetLoadingTask()
        result = task.run(context)

        # Verify system prompt was loaded from dataset.json
        assert "system_prompt" in result
        assert (
            result["system_prompt"]
            == "You are an expert at identifying materials. Return JSON format."
        )

        # Verify it's also in config for VLMInferenceTask
        assert "config" in result
        assert "system_prompt" in result["config"]
        assert result["config"]["system_prompt"] == result["system_prompt"]

        # Verify dataset was loaded
        assert "dataset" in result
        assert len(result["dataset"]) == 1

    def test_dataset_loading_respects_existing_system_prompt(self, tmp_path):
        """Test that DatasetLoadingTask doesn't override existing system prompt."""
        from material_agent.tasks.dataset import DatasetLoadingTask

        # Setup: Create minimal dataset
        dataset_jsonl = tmp_path / "dataset.jsonl"
        dataset_entry = {
            "id": "test_entry",
            "media": {"images": [{"path": "image1.png"}]},
            "user_prompt": "Test",
        }

        with open(dataset_jsonl, "w") as f:
            f.write(json.dumps(dataset_entry) + "\n")

        # Create dataset.json with system prompt
        dataset_json = tmp_path / "dataset.json"
        with open(dataset_json, "w") as f:
            json.dump(
                {
                    "inference": {
                        "prompts": [{"system_prompt": "System prompt from dataset"}]
                    }
                },
                f,
            )

        # Create test image
        image1 = tmp_path / "image1.png"
        from PIL import Image

        Image.new("RGB", (100, 100)).save(image1)

        # Setup: Context with existing system_prompt
        context = {
            "dataset_path": str(dataset_jsonl),
            "system_prompt": "Existing system prompt from config",
            "config": {"system_prompt": "Existing system prompt from config"},
        }

        # Run task
        task = DatasetLoadingTask()
        result = task.run(context)

        # Verify existing system prompt was NOT overridden
        assert result["system_prompt"] == "Existing system prompt from config"
        assert result["config"]["system_prompt"] == "Existing system prompt from config"


def _create_input_usd(path: Path, default_prim: str | None = "RootNode") -> None:
    """Helper to create a minimal valid USD file with a defaultPrim and mesh."""
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    if default_prim:
        root = UsdGeom.Xform.Define(stage, f"/{default_prim}")
        stage.SetDefaultPrim(root.GetPrim())
        UsdGeom.Scope.Define(stage, f"/{default_prim}/Geometry")
        UsdGeom.Mesh.Define(stage, f"/{default_prim}/Geometry/Part1")
    stage.GetRootLayer().Save()


def _create_predictions(path: Path, prim_prefix: str = "/RootNode") -> None:
    """Helper to create a minimal predictions JSONL file."""
    predictions = [
        {
            "id": f"{prim_prefix}/Geometry/Part1",
            "materials": {"material": "TestMaterial"},
        }
    ]
    with open(path, "w") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")


def _create_material_library(path: Path) -> None:
    """Helper to create a minimal material library USD with one material."""
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.Scope.Define(stage, "/World/Looks")
    UsdShade.Material.Define(stage, "/World/Looks/TestMaterial")
    stage.GetRootLayer().Save()


def _define_openpbr_material(stage: Usd.Stage, material_path: str) -> UsdShade.Material:
    """Define a minimal OpenPBR/MaterialX material in an existing stage."""
    UsdGeom.Scope.Define(stage, str(Sdf.Path(material_path).GetParentPath()))
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(
        stage,
        f"{material_path}/open_pbr_surface_surfaceshader",
    )
    shader.CreateIdAttr("ND_open_pbr_surface_surfaceshader")
    shader.CreateInput("base_color", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.2, 0.4, 0.8)
    )
    shader.CreateInput("specular_roughness", Sdf.ValueTypeNames.Float).Set(0.35)
    shader.CreateInput("base_metalness", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput("mtlx").ConnectToSource(
        shader.ConnectableAPI(),
        "out",
    )
    material.CreateSurfaceOutput()
    return material


def _create_openpbr_material_library(path: Path) -> None:
    """Create a minimal OpenPBR/MaterialX material library."""
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.Scope.Define(stage, "/World/Looks")
    _define_openpbr_material(stage, "/World/Looks/TestMaterial")
    stage.GetRootLayer().Save()


def _create_omnipbr_mdl_material_library(path: Path) -> None:
    """Create a minimal OmniPBR MDL material library."""
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.Scope.Define(stage, "/World/Looks")

    material = UsdShade.Material.Define(stage, "/World/Looks/TestMaterial")
    shader = UsdShade.Shader.Define(stage, "/World/Looks/TestMaterial/Shader")
    shader_prim = shader.GetPrim()
    shader_prim.CreateAttribute(
        "info:mdl:sourceAsset",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath("OmniPBR.mdl"))
    shader_prim.CreateAttribute(
        "info:mdl:sourceAsset:subIdentifier",
        Sdf.ValueTypeNames.Token,
    ).Set("OmniPBR")
    material.CreateSurfaceOutput("mdl").ConnectToSource(
        shader.ConnectableAPI(),
        "out",
    )
    stage.GetRootLayer().Save()


def _add_bound_geom_subset(path: Path) -> None:
    """Add a GeomSubset with its own material binding under the test mesh."""
    stage = Usd.Stage.Open(str(path))
    material = UsdShade.Material.Define(stage, "/RootNode/Looks/SubsetMaterial")
    subset = UsdGeom.Subset.Define(stage, "/RootNode/Geometry/Part1/Subset0")
    subset.CreateElementTypeAttr(UsdGeom.Tokens.face)
    subset.CreateFamilyNameAttr("materialBind")
    subset.CreateIndicesAttr([0])
    UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()


class TestMaterialProfileResolution:
    """Tests for explicit material authoring profile behavior."""

    def test_auto_preserves_file_mdl_profile(self, tmp_path: Path) -> None:
        input_usd = tmp_path / "input.usda"
        output_usd = tmp_path / "output.usda"
        predictions_path = tmp_path / "predictions.jsonl"
        _create_input_usd(input_usd)
        _create_predictions(predictions_path)

        result = ApplyMaterialsToUSDTask().run(
            {
                "input_usd_path": str(input_usd),
                "output_usd_path": str(output_usd),
                "predictions_path": str(predictions_path),
                "resolved_materials": {"TestMaterial": str(tmp_path / "test.mdl")},
                "flatten_output": False,
            }
        )

        profile = result["material_profile_result"]
        assert profile["requested_profile"] == "auto"
        assert profile["resolved_profile"] == "omnipbr_mdl"
        assert profile["materials"]["TestMaterial"]["source_profile"] == "omnipbr_mdl"

        stage = Usd.Stage.Open(str(output_usd))
        shader = stage.GetPrimAtPath("/RootNode/Looks/TestMaterial/Shader")
        assert (
            shader.GetAttribute("info:mdl:sourceAsset").Get().path.endswith("test.mdl")
        )

    def test_explicit_preview_surface_file_material_does_not_author_mdl(
        self, tmp_path: Path
    ) -> None:
        input_usd = tmp_path / "input.usda"
        output_usd = tmp_path / "output.usda"
        predictions_path = tmp_path / "predictions.jsonl"
        _create_input_usd(input_usd)
        _create_predictions(predictions_path)

        result = ApplyMaterialsToUSDTask().run(
            {
                "input_usd_path": str(input_usd),
                "output_usd_path": str(output_usd),
                "predictions_path": str(predictions_path),
                "resolved_materials": {"TestMaterial": str(tmp_path / "test.mdl")},
                "material_profile": "preview_surface",
                "flatten_output": False,
            }
        )

        profile = result["material_profile_result"]
        assert profile["requested_profile"] == "preview_surface"
        assert profile["resolved_profile"] == "preview_surface"
        assert (
            profile["materials"]["TestMaterial"]["fallback_reason"]
            == "explicit_preview_surface_requested_for_mdl"
        )
        assert (
            profile["warnings"][0]["code"] == "material_profile.mdl_to_preview_surface"
        )

        stage = Usd.Stage.Open(str(output_usd))
        shader_ids = [
            prim.GetAttribute("info:id").Get()
            for prim in stage.Traverse()
            if prim.IsA(UsdShade.Shader)
        ]
        assert shader_ids == ["UsdPreviewSurface"]
        assert not any(
            prim.GetAttribute("info:mdl:sourceAsset")
            and prim.GetAttribute("info:mdl:sourceAsset").Get()
            for prim in stage.Traverse()
            if prim.IsA(UsdShade.Shader)
        )

    def test_explicit_openpbr_rejects_file_mdl_material(self, tmp_path: Path) -> None:
        input_usd = tmp_path / "input.usda"
        output_usd = tmp_path / "output.usda"
        predictions_path = tmp_path / "predictions.jsonl"
        _create_input_usd(input_usd)
        _create_predictions(predictions_path)

        with pytest.raises(ValueError, match="openpbr_materialx was requested"):
            ApplyMaterialsToUSDTask().run(
                {
                    "input_usd_path": str(input_usd),
                    "output_usd_path": str(output_usd),
                    "predictions_path": str(predictions_path),
                    "resolved_materials": {"TestMaterial": str(tmp_path / "test.mdl")},
                    "material_profile": "openpbr_materialx",
                    "flatten_output": False,
                }
            )
        assert not output_usd.exists()

    def test_explicit_display_color_authors_primvars_without_material_binding(
        self, tmp_path: Path
    ) -> None:
        input_usd = tmp_path / "input.usda"
        output_usd = tmp_path / "output.usda"
        predictions_path = tmp_path / "predictions.jsonl"
        _create_input_usd(input_usd)
        _create_predictions(predictions_path)

        result = ApplyMaterialsToUSDTask().run(
            {
                "input_usd_path": str(input_usd),
                "output_usd_path": str(output_usd),
                "predictions_path": str(predictions_path),
                "resolved_materials": {"TestMaterial": str(tmp_path / "test.mdl")},
                "material_profile": "display_color",
                "flatten_output": False,
            }
        )

        profile = result["material_profile_result"]
        assert profile["resolved_profile"] == "display_color"
        assert profile["errors"] == []
        assert profile["materials"]["TestMaterial"]["authoring_backend"] == (
            "usd.primvars.displayColor"
        )
        assert profile["authoring_backends"] == ["usd.primvars.displayColor"]

        stage = Usd.Stage.Open(str(output_usd))
        prim = stage.GetPrimAtPath("/RootNode/Geometry/Part1")
        display_color = UsdGeom.PrimvarsAPI(prim).GetPrimvar("displayColor")
        display_opacity = UsdGeom.PrimvarsAPI(prim).GetPrimvar("displayOpacity")
        assert list(display_color.Get()[0]) == pytest.approx([0.5, 0.5, 0.5])
        assert list(display_opacity.Get()) == pytest.approx([1.0])
        bound_material, _relationship = UsdShade.MaterialBindingAPI(
            prim
        ).ComputeBoundMaterial()
        assert not bound_material
        assert not stage.GetPrimAtPath("/RootNode/Looks/TestMaterial")

    @pytest.mark.parametrize("layer_only", [False, True])
    def test_assignment_stats_include_every_prediction_target(
        self,
        tmp_path: Path,
        layer_only: bool,
    ) -> None:
        input_usd = tmp_path / "input.usda"
        output_usd = tmp_path / f"output-{layer_only}.usda"
        predictions_path = tmp_path / "predictions.jsonl"
        _create_input_usd(input_usd)
        stage = Usd.Stage.Open(str(input_usd))
        for name in (
            "Default",
            "Missing",
            "Fallback",
            "KeyedDefault",
            "KeyedEmpty",
            "KeyedMissing",
            "KeyedNull",
        ):
            UsdGeom.Mesh.Define(stage, f"/RootNode/Geometry/{name}")
        stage.GetRootLayer().Save()
        predictions_path.write_text(
            "\n".join(
                json.dumps(record)
                for record in (
                    {
                        "id": "/RootNode/Geometry/Part1",
                        "materials": {"material": "Steel"},
                    },
                    {
                        "id": "/RootNode/Geometry/Default",
                        "materials": {"material": "__USE_DEFAULT_LIBRARY__"},
                    },
                    {"id": "/RootNode/Geometry/Missing"},
                    {
                        "id": "/RootNode/Geometry/Fallback",
                        "materials": {"material": "__UNKNOWN__"},
                    },
                    {
                        "/RootNode/Geometry/KeyedDefault": {
                            "materials": {"material": "__USE_DEFAULT_LIBRARY__"}
                        },
                        "/RootNode/Geometry/KeyedEmpty": [],
                        "/RootNode/Geometry/KeyedMissing": {},
                        "/RootNode/Geometry/KeyedNull": None,
                    },
                )
            )
            + "\n",
            encoding="utf-8",
        )

        result = ApplyMaterialsToUSDTask().run(
            {
                "input_usd_path": str(input_usd),
                "output_usd_path": str(output_usd),
                "predictions_path": str(predictions_path),
                "resolved_materials": {"Steel": "unused"},
                "material_profile": "display_color",
                "layer_only": layer_only,
            }
        )

        assignment_stats = result["assignment_stats"]
        assert set(assignment_stats["bound_prim_ids"]) == {
            "/RootNode/Geometry/Fallback",
            "/RootNode/Geometry/Part1",
        }
        assert assignment_stats["unbound_prim_ids"] == [
            "/RootNode/Geometry/Default",
            "/RootNode/Geometry/KeyedDefault",
            "/RootNode/Geometry/KeyedEmpty",
            "/RootNode/Geometry/KeyedMissing",
            "/RootNode/Geometry/KeyedNull",
            "/RootNode/Geometry/Missing",
        ]

    def test_explicit_display_color_layer_only_blocks_weaker_binding(
        self, tmp_path: Path
    ) -> None:
        input_usd = tmp_path / "input.usda"
        output_usd = tmp_path / "display_layer.usda"
        predictions_path = tmp_path / "predictions.jsonl"
        _create_input_usd(input_usd)
        _add_bound_geom_subset(input_usd)
        _create_predictions(predictions_path)

        result = ApplyMaterialsToUSDTask().run(
            {
                "input_usd_path": str(input_usd),
                "output_usd_path": str(output_usd),
                "predictions_path": str(predictions_path),
                "resolved_materials": {"TestMaterial": str(tmp_path / "test.mdl")},
                "material_profile": "display_color",
                "layer_only": True,
            }
        )

        assert result["resolved_material_profile"] == "display_color"
        stage = Usd.Stage.Open(str(output_usd))
        prim = stage.GetPrimAtPath("/RootNode/Geometry/Part1")
        assert list(
            UsdGeom.PrimvarsAPI(prim).GetPrimvar("displayColor").Get()[0]
        ) == pytest.approx([0.5, 0.5, 0.5])
        bound_material, _relationship = UsdShade.MaterialBindingAPI(
            prim
        ).ComputeBoundMaterial()
        assert not bound_material

        prim_spec = stage.GetRootLayer().GetPrimAtPath("/RootNode/Geometry/Part1")
        binding_rel = prim_spec.relationships["material:binding"]
        assert list(binding_rel.targetPathList.explicitItems) == []

        subset_prim = stage.GetPrimAtPath("/RootNode/Geometry/Part1/Subset0")
        assert list(
            UsdGeom.PrimvarsAPI(subset_prim).GetPrimvar("displayColor").Get()[0]
        ) == pytest.approx([0.5, 0.5, 0.5])
        subset_material, _relationship = UsdShade.MaterialBindingAPI(
            subset_prim
        ).ComputeBoundMaterial()
        assert not subset_material
        subset_spec = stage.GetRootLayer().GetPrimAtPath(
            "/RootNode/Geometry/Part1/Subset0"
        )
        subset_binding_rel = subset_spec.relationships["material:binding"]
        assert list(subset_binding_rel.targetPathList.explicitItems) == []

    def test_library_auto_preserves_openpbr_without_preview_fallback(
        self, tmp_path: Path
    ) -> None:
        input_usd = tmp_path / "input.usda"
        output_usd = tmp_path / "output.usda"
        predictions_path = tmp_path / "predictions.jsonl"
        library_path = tmp_path / "library.usda"
        _create_input_usd(input_usd)
        _create_predictions(predictions_path)
        _create_openpbr_material_library(library_path)

        result = ApplyMaterialsToUSDTask().run(
            {
                "input_usd_path": str(input_usd),
                "output_usd_path": str(output_usd),
                "predictions_path": str(predictions_path),
                "resolved_materials": {"TestMaterial": "/World/Looks/TestMaterial"},
                "is_library_based_mapping": True,
                "material_library_path": str(library_path),
                "flatten_output": False,
            }
        )

        profile = result["material_profile_result"]
        assert profile["requested_profile"] == "auto"
        assert profile["resolved_profile"] == "openpbr_materialx"
        assert profile["warnings"] == []

        stage = Usd.Stage.Open(str(output_usd))
        material = UsdShade.Material(
            stage.GetPrimAtPath("/RootNode/Looks/TestMaterial")
        )
        sources, _ = material.GetSurfaceOutput("mtlx").GetConnectedSources()
        assert sources
        assert not stage.GetPrimAtPath(
            "/RootNode/Looks/TestMaterial/OVRTXPreviewSurface"
        )

    def test_explicit_preview_surface_adds_library_openpbr_overlay(
        self, tmp_path: Path
    ) -> None:
        input_usd = tmp_path / "input.usda"
        output_usd = tmp_path / "output.usda"
        predictions_path = tmp_path / "predictions.jsonl"
        library_path = tmp_path / "library.usda"
        _create_input_usd(input_usd)
        _create_predictions(predictions_path)
        _create_openpbr_material_library(library_path)

        result = ApplyMaterialsToUSDTask().run(
            {
                "input_usd_path": str(input_usd),
                "output_usd_path": str(output_usd),
                "predictions_path": str(predictions_path),
                "resolved_materials": {"TestMaterial": "/World/Looks/TestMaterial"},
                "is_library_based_mapping": True,
                "material_library_path": str(library_path),
                "material_profile": "preview_surface",
                "flatten_output": False,
            }
        )

        profile = result["material_profile_result"]
        assert profile["resolved_profile"] == "preview_surface"
        assert (
            profile["materials"]["TestMaterial"]["fallback_reason"]
            == "explicit_preview_surface_requested_for_openpbr_materialx"
        )
        assert (
            profile["warnings"][0]["code"] == "material_profile.preview_surface_overlay"
        )

        stage = Usd.Stage.Open(str(output_usd))
        material = UsdShade.Material(
            stage.GetPrimAtPath("/RootNode/Looks/TestMaterial")
        )
        mtlx_sources, _ = material.GetSurfaceOutput("mtlx").GetConnectedSources()
        assert mtlx_sources == []
        preview_shader = stage.GetPrimAtPath(
            "/RootNode/Looks/TestMaterial/OVRTXPreviewSurface"
        )
        assert preview_shader.GetAttribute("info:id").Get() == "UsdPreviewSurface"

    def test_explicit_preview_surface_adds_library_mdl_overlay(
        self, tmp_path: Path
    ) -> None:
        input_usd = tmp_path / "input.usda"
        output_usd = tmp_path / "output.usda"
        predictions_path = tmp_path / "predictions.jsonl"
        library_path = tmp_path / "library.usda"
        _create_input_usd(input_usd)
        _create_predictions(predictions_path)
        _create_omnipbr_mdl_material_library(library_path)

        result = ApplyMaterialsToUSDTask().run(
            {
                "input_usd_path": str(input_usd),
                "output_usd_path": str(output_usd),
                "predictions_path": str(predictions_path),
                "resolved_materials": {"TestMaterial": "/World/Looks/TestMaterial"},
                "is_library_based_mapping": True,
                "material_library_path": str(library_path),
                "material_profile": "preview_surface",
                "flatten_output": False,
            }
        )

        profile = result["material_profile_result"]
        assert profile["resolved_profile"] == "preview_surface"
        assert (
            profile["materials"]["TestMaterial"]["fallback_reason"]
            == "explicit_preview_surface_requested_for_mdl"
        )
        assert (
            profile["warnings"][0]["code"] == "material_profile.mdl_to_preview_surface"
        )

        stage = Usd.Stage.Open(str(output_usd))
        material = UsdShade.Material(
            stage.GetPrimAtPath("/RootNode/Looks/TestMaterial")
        )
        assert material.GetSurfaceOutput().HasConnectedSource()
        mdl_sources, _ = material.GetSurfaceOutput("mdl").GetConnectedSources()
        assert mdl_sources == []
        preview_shader = stage.GetPrimAtPath(
            "/RootNode/Looks/TestMaterial/PreviewSurface"
        )
        assert preview_shader.GetAttribute("info:id").Get() == "UsdPreviewSurface"


class TestDefaultPrimPreservation:
    """Tests that defaultPrim is preserved from input to output.

    defaultPrim is non-composable USD layer metadata — it only takes effect on
    the root layer and does not compose from sublayers. Both _create_full_stage()
    and _create_material_layer() must explicitly copy it from the input.
    """

    def test_full_stage_preserves_default_prim(self, tmp_path):
        """_create_full_stage() must copy defaultPrim from input to output."""
        input_usd = tmp_path / "input.usda"
        output_usd = tmp_path / "output.usda"
        predictions_path = tmp_path / "predictions.jsonl"
        library_path = tmp_path / "library.usda"

        _create_input_usd(input_usd, default_prim="RootNode")
        _create_predictions(predictions_path, prim_prefix="/RootNode")
        _create_material_library(library_path)

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        stage, materials_applied, stats = task._create_full_stage(
            input_usd_path=input_usd,
            output_usd_path=output_usd,
            resolved_materials={"TestMaterial": "/World/Looks/TestMaterial"},
            prim_to_material={"/RootNode/Geometry/Part1": "TestMaterial"},
            is_library_based=True,
            material_library_path=str(library_path),
            flatten_output=False,
        )

        # Verify defaultPrim is set on the output root layer
        output_layer = Sdf.Layer.FindOrOpen(str(output_usd))
        assert output_layer.defaultPrim == "RootNode"

        # Verify via composed stage
        output_stage = Usd.Stage.Open(str(output_usd))
        assert output_stage.HasDefaultPrim()
        assert str(output_stage.GetDefaultPrim().GetPath()) == "/RootNode"

        # Library-based full-stage output authors bindings directly on the
        # output root layer, avoiding native binding helpers for library graphs.
        bound_material, _binding_rel = UsdShade.MaterialBindingAPI(
            output_stage.GetPrimAtPath("/RootNode/Geometry/Part1")
        ).ComputeBoundMaterial()
        assert str(bound_material.GetPrim().GetPath()) == "/RootNode/Looks/TestMaterial"

        prim_spec = output_layer.GetPrimAtPath("/RootNode/Geometry/Part1")
        assert prim_spec.specifier == Sdf.SpecifierOver
        assert prim_spec.relationships[
            "material:binding"
        ].targetPathList.explicitItems == [Sdf.Path("/RootNode/Looks/TestMaterial")]

    def test_sdf_binding_preserves_existing_api_schemas(self, tmp_path):
        """Sdf-level binding must not discard existing API schemas."""
        output_layer = Sdf.Layer.CreateNew(str(tmp_path / "output.usda"))
        prim_spec = Sdf.CreatePrimInLayer(output_layer, "/RootNode/Geometry/Part1")
        prim_spec.specifier = Sdf.SpecifierOver
        prim_spec.SetInfo(
            "apiSchemas",
            Sdf.TokenListOp.Create(prependedItems=["PhysicsRigidBodyAPI"]),
        )

        task = ApplyMaterialsToUSDTask()
        task._author_material_binding_in_layer(
            output_layer,
            "/RootNode/Geometry/Part1",
            "/RootNode/Looks/TestMaterial",
        )

        api_schemas = prim_spec.GetInfo("apiSchemas")
        assert "MaterialBindingAPI" in api_schemas.prependedItems
        assert "PhysicsRigidBodyAPI" in api_schemas.prependedItems
        assert prim_spec.relationships[
            "material:binding"
        ].targetPathList.explicitItems == [Sdf.Path("/RootNode/Looks/TestMaterial")]

    def test_sdf_binding_preserves_existing_def_specifier(self, tmp_path):
        """Sdf-level binding must not turn existing geometry defs into overs."""
        output_layer = Sdf.Layer.CreateNew(str(tmp_path / "output.usda"))
        prim_spec = Sdf.CreatePrimInLayer(output_layer, "/RootNode/Geometry/Part1")
        prim_spec.specifier = Sdf.SpecifierDef
        prim_spec.typeName = "Mesh"

        task = ApplyMaterialsToUSDTask()
        task._author_material_binding_in_layer(
            output_layer,
            "/RootNode/Geometry/Part1",
            "/RootNode/Looks/TestMaterial",
        )

        assert prim_spec.specifier == Sdf.SpecifierDef
        assert prim_spec.typeName == "Mesh"
        assert prim_spec.relationships[
            "material:binding"
        ].targetPathList.explicitItems == [Sdf.Path("/RootNode/Looks/TestMaterial")]

    def test_full_stage_preserves_different_default_prim_name(self, tmp_path):
        """_create_full_stage() works with non-standard defaultPrim names."""
        input_usd = tmp_path / "input.usda"
        output_usd = tmp_path / "output.usda"
        predictions_path = tmp_path / "predictions.jsonl"
        library_path = tmp_path / "library.usda"

        _create_input_usd(input_usd, default_prim="World")
        _create_predictions(predictions_path, prim_prefix="/World")
        _create_material_library(library_path)

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        task._create_full_stage(
            input_usd_path=input_usd,
            output_usd_path=output_usd,
            resolved_materials={"TestMaterial": "/World/Looks/TestMaterial"},
            prim_to_material={"/World/Geometry/Part1": "TestMaterial"},
            is_library_based=True,
            material_library_path=str(library_path),
            flatten_output=False,
        )

        output_layer = Sdf.Layer.FindOrOpen(str(output_usd))
        assert output_layer.defaultPrim == "World"

    def test_full_stage_handles_no_default_prim(self, tmp_path):
        """_create_full_stage() auto-detects root prim when input has no defaultPrim."""
        input_usd = tmp_path / "input.usda"
        output_usd = tmp_path / "output.usda"
        predictions_path = tmp_path / "predictions.jsonl"
        library_path = tmp_path / "library.usda"

        # Create input WITHOUT a defaultPrim
        _create_input_usd(input_usd, default_prim=None)
        # Need a prim so the stage is valid — add one manually
        stage = Usd.Stage.Open(str(input_usd))
        UsdGeom.Xform.Define(stage, "/SomeRoot")
        UsdGeom.Mesh.Define(stage, "/SomeRoot/Mesh")
        stage.GetRootLayer().Save()

        _create_predictions(predictions_path, prim_prefix="/SomeRoot")
        _create_material_library(library_path)

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        task._create_full_stage(
            input_usd_path=input_usd,
            output_usd_path=output_usd,
            resolved_materials={"TestMaterial": "/World/Looks/TestMaterial"},
            prim_to_material={"/SomeRoot/Mesh": "TestMaterial"},
            is_library_based=True,
            material_library_path=str(library_path),
            flatten_output=False,
        )

        # When input has no defaultPrim, the fix auto-detects the actual root
        # prim from the composed stage so materials are placed correctly.
        output_layer = Sdf.Layer.FindOrOpen(str(output_usd))
        assert output_layer.defaultPrim == "SomeRoot"

    def test_material_layer_preserves_default_prim(self, tmp_path):
        """_create_material_layer() must copy defaultPrim from input to output."""
        input_usd = tmp_path / "input.usda"
        output_usd = tmp_path / "output.usda"
        predictions_path = tmp_path / "predictions.jsonl"
        library_path = tmp_path / "library.usda"

        _create_input_usd(input_usd, default_prim="RootNode")
        _create_predictions(predictions_path, prim_prefix="/RootNode")
        _create_material_library(library_path)

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        stage, materials_applied, stats = task._create_material_layer(
            input_usd_path=input_usd,
            output_usd_path=output_usd,
            resolved_materials={"TestMaterial": "/World/Looks/TestMaterial"},
            prim_to_material={"/RootNode/Geometry/Part1": "TestMaterial"},
            is_library_based=True,
            material_library_path=str(library_path),
        )

        # Verify defaultPrim is set on the output root layer
        output_layer = Sdf.Layer.FindOrOpen(str(output_usd))
        assert output_layer.defaultPrim == "RootNode"

        # Verify via composed stage
        output_stage = Usd.Stage.Open(str(output_usd))
        assert output_stage.HasDefaultPrim()
        assert str(output_stage.GetDefaultPrim().GetPath()) == "/RootNode"

    def test_up_axis_also_preserved(self, tmp_path):
        """Verify upAxis is preserved alongside defaultPrim."""
        input_usd = tmp_path / "input.usda"
        output_usd = tmp_path / "output.usda"
        predictions_path = tmp_path / "predictions.jsonl"
        library_path = tmp_path / "library.usda"

        _create_input_usd(input_usd, default_prim="RootNode")
        _create_predictions(predictions_path, prim_prefix="/RootNode")
        _create_material_library(library_path)

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        task._create_full_stage(
            input_usd_path=input_usd,
            output_usd_path=output_usd,
            resolved_materials={"TestMaterial": "/World/Looks/TestMaterial"},
            prim_to_material={"/RootNode/Geometry/Part1": "TestMaterial"},
            is_library_based=True,
            material_library_path=str(library_path),
            flatten_output=False,
        )

        output_stage = Usd.Stage.Open(str(output_usd))
        assert UsdGeom.GetStageUpAxis(output_stage) == UsdGeom.Tokens.z

    def test_full_stage_fixes_stale_default_prim(self, tmp_path):
        """_create_full_stage() corrects stale defaultPrim after optimizer renames root.

        When the NVCF optimizer wraps content under /World but the input's
        defaultPrim still says 'OriginalRoot', the composed stage has no valid
        default prim. The fix detects this and updates defaultPrim to match the
        actual root, so materials go under the correct prim.
        """
        input_usd = tmp_path / "input.usda"
        output_usd = tmp_path / "output.usda"
        predictions_path = tmp_path / "predictions.jsonl"
        library_path = tmp_path / "library.usda"

        # Create input simulating NVCF optimizer output:
        # - Content under /World (optimizer's convention)
        # - But defaultPrim still says "OriginalRoot" (stale from pre-optimization)
        stage = Usd.Stage.CreateNew(str(input_usd))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.Xform.Define(stage, "/World")
        UsdGeom.Scope.Define(stage, "/World/Geometry")
        UsdGeom.Mesh.Define(stage, "/World/Geometry/Part1")
        stage.GetRootLayer().defaultPrim = "OriginalRoot"  # Stale!
        stage.GetRootLayer().Save()

        _create_predictions(predictions_path, prim_prefix="/World")
        _create_material_library(library_path)

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        task._create_full_stage(
            input_usd_path=input_usd,
            output_usd_path=output_usd,
            resolved_materials={"TestMaterial": "/World/Looks/TestMaterial"},
            prim_to_material={"/World/Geometry/Part1": "TestMaterial"},
            is_library_based=True,
            material_library_path=str(library_path),
            flatten_output=False,
        )

        # Verify defaultPrim was corrected to the actual root prim
        output_layer = Sdf.Layer.FindOrOpen(str(output_usd))
        assert output_layer.defaultPrim == "World"

        # Verify via composed stage
        output_stage = Usd.Stage.Open(str(output_usd))
        assert output_stage.HasDefaultPrim()
        assert str(output_stage.GetDefaultPrim().GetPath()) == "/World"


class TestApplyMaterialsOutputIntegrity:
    """Regression tests for output USD integrity (metersPerUnit, no extra prims)."""

    def test_flatten_preserves_meters_per_unit(self, tmp_path):
        """Flatten must not change metersPerUnit from the original stage.

        Regression: flatten was silently resetting metersPerUnit to 0.01
        when the original asset used 1.0 (meters).
        """
        input_usd = tmp_path / "input.usda"
        output_usd = tmp_path / "output.usd"

        # Create input with metersPerUnit=1.0 (meters, NOT the 0.01 default)
        stage = Usd.Stage.CreateNew(str(input_usd))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        root = UsdGeom.Xform.Define(stage, "/Asset")
        stage.SetDefaultPrim(root.GetPrim())
        UsdGeom.Mesh.Define(stage, "/Asset/Mesh")
        stage.GetRootLayer().Save()

        # Verify input
        assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        # Run with flatten_output=True (the default in the service)
        task._create_full_stage(
            input_usd_path=input_usd,
            output_usd_path=output_usd,
            resolved_materials={},
            prim_to_material={},
            flatten_output=True,
        )

        # Verify metersPerUnit is preserved
        out_stage = Usd.Stage.Open(str(output_usd))
        assert UsdGeom.GetStageMetersPerUnit(out_stage) == 1.0, (
            f"metersPerUnit changed from 1.0 to "
            f"{UsdGeom.GetStageMetersPerUnit(out_stage)} after flatten"
        )

    def test_flatten_preserves_up_axis(self, tmp_path):
        """Flatten must preserve the original upAxis."""
        input_usd = tmp_path / "input.usda"
        output_usd = tmp_path / "output.usd"

        stage = Usd.Stage.CreateNew(str(input_usd))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        root = UsdGeom.Xform.Define(stage, "/Asset")
        stage.SetDefaultPrim(root.GetPrim())
        stage.GetRootLayer().Save()

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        task._create_full_stage(
            input_usd_path=input_usd,
            output_usd_path=output_usd,
            resolved_materials={},
            prim_to_material={},
            flatten_output=True,
        )

        out_stage = Usd.Stage.Open(str(output_usd))
        assert UsdGeom.GetStageUpAxis(out_stage) == UsdGeom.Tokens.z

    def test_flatten_continues_when_stale_shader_cleanup_fails(
        self, monkeypatch, tmp_path
    ):
        """Flatten should still export if best-effort stale shader cleanup fails."""
        input_usd = tmp_path / "input.usda"
        output_usd = tmp_path / "output.usd"

        stage = Usd.Stage.CreateNew(str(input_usd))
        root = UsdGeom.Xform.Define(stage, "/Asset")
        stage.SetDefaultPrim(root.GetPrim())
        UsdGeom.Mesh.Define(stage, "/Asset/Mesh")
        stage.GetRootLayer().Save()

        def fail_cleanup(*args, **kwargs):
            raise RuntimeError("cleanup failed")

        monkeypatch.setattr(
            ApplyMaterialsToUSDTask,
            "_deactivate_unbound_unresolved_mdl_shaders",
            fail_cleanup,
        )

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        task._create_full_stage(
            input_usd_path=input_usd,
            output_usd_path=output_usd,
            resolved_materials={},
            prim_to_material={},
            flatten_output=True,
        )

        assert output_usd.exists()
        warnings = [call.args[0] for call in task.listener.warning.call_args_list]
        assert any(
            "Failed to deactivate stale unresolved MDL shaders" in warning
            for warning in warnings
        )

    def test_layer_only_has_no_geometry(self, tmp_path):
        """layer_only output must not contain geometry from the input."""
        input_usd = tmp_path / "input.usda"
        output_usd = tmp_path / "output.usd"

        stage = Usd.Stage.CreateNew(str(input_usd))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        root = UsdGeom.Xform.Define(stage, "/Asset")
        stage.SetDefaultPrim(root.GetPrim())
        UsdGeom.Mesh.Define(stage, "/Asset/Body")
        UsdGeom.Mesh.Define(stage, "/Asset/Wheel")
        stage.GetRootLayer().Save()

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        task._create_material_layer(
            input_usd_path=input_usd,
            output_usd_path=output_usd,
            resolved_materials={},
            prim_to_material={},
        )

        # The output root layer should NOT define the geometry prims
        # (they come through sublayer composition, not the root layer)
        out_layer = Sdf.Layer.FindOrOpen(str(output_usd))
        root_prims = [p.name for p in out_layer.rootPrims]
        assert "Asset" not in root_prims or all(
            out_layer.GetPrimAtPath(f"/Asset/{child}").specifier == Sdf.SpecifierOver
            for child in ["Body", "Wheel"]
            if out_layer.GetPrimAtPath(f"/Asset/{child}")
        ), "layer_only output should use 'over' specs, not 'def' for geometry"

    def test_library_materials_placed_under_default_prim(self, tmp_path):
        """Library materials must go under the asset's defaultPrim, not /World.

        Regression: materials from the library at /World/Looks/Iron were
        copied verbatim, creating an extra /World root prim in the output.
        """
        input_usd = tmp_path / "input.usda"
        output_usd = tmp_path / "output.usd"
        library_usd = tmp_path / "library.usd"

        # Create input with defaultPrim = "MyGear"
        stage = Usd.Stage.CreateNew(str(input_usd))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
        root = UsdGeom.Xform.Define(stage, "/MyGear")
        stage.SetDefaultPrim(root.GetPrim())
        UsdGeom.Mesh.Define(stage, "/MyGear/Body")
        stage.GetRootLayer().Save()

        # Create library with materials under /World/Looks
        lib_stage = Usd.Stage.CreateNew(str(library_usd))
        UsdGeom.Scope.Define(lib_stage, "/World")
        UsdGeom.Scope.Define(lib_stage, "/World/Looks")
        UsdShade.Material.Define(lib_stage, "/World/Looks/Iron")
        lib_stage.GetRootLayer().Save()

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        task._create_full_stage(
            input_usd_path=input_usd,
            output_usd_path=output_usd,
            resolved_materials={"Iron": "/World/Looks/Iron"},
            prim_to_material={"/MyGear/Body": "Iron"},
            is_library_based=True,
            material_library_path=str(library_usd),
            flatten_output=True,
        )

        out_stage = Usd.Stage.Open(str(output_usd))
        root_prims = [p.GetName() for p in out_stage.GetPseudoRoot().GetChildren()]

        # /World must NOT be a root prim — materials should be under /MyGear
        assert "World" not in root_prims, (
            f"Output has /World root prim — materials should be under "
            f"the default prim /MyGear. Root prims: {root_prims}"
        )

        # Materials should be under /MyGear/Looks/Iron
        iron_prim = out_stage.GetPrimAtPath("/MyGear/Looks/Iron")
        assert iron_prim.IsValid(), (
            "Material should be at /MyGear/Looks/Iron, not /World/Looks/Iron"
        )
        looks_prim = out_stage.GetPrimAtPath("/MyGear/Looks")
        assert looks_prim.IsA(UsdGeom.Scope)

    def test_library_copy_clears_color_space_on_empty_asset_inputs(self, tmp_path):
        """Empty texture slots copied from a library must not keep colorSpace."""
        input_usd = tmp_path / "input.usda"
        output_usd = tmp_path / "output.usd"
        library_usd = tmp_path / "library.usd"

        stage = Usd.Stage.CreateNew(str(input_usd))
        root = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(root.GetPrim())
        UsdGeom.Mesh.Define(stage, "/World/Body")
        stage.GetRootLayer().Save()

        lib_stage = Usd.Stage.CreateNew(str(library_usd))
        UsdGeom.Scope.Define(lib_stage, "/World")
        UsdGeom.Scope.Define(lib_stage, "/World/Looks")
        material = UsdShade.Material.Define(lib_stage, "/World/Looks/Steel")
        material_prim = material.GetPrim()
        empty_texture = material_prim.CreateAttribute(
            "inputs:base_color_texture_file",
            Sdf.ValueTypeNames.Asset,
        )
        empty_texture.Set(Sdf.AssetPath(""))
        empty_texture.SetColorSpace("sRGB")
        real_texture = material_prim.CreateAttribute(
            "inputs:geometry_normal_texture_file",
            Sdf.ValueTypeNames.Asset,
        )
        real_texture.Set(Sdf.AssetPath("textures/normal.png"))
        real_texture.SetColorSpace("raw")
        shader = UsdShade.Shader.Define(lib_stage, "/World/Looks/Steel/Shader")
        nested_empty_texture = shader.CreateInput(
            "emissive_texture_file",
            Sdf.ValueTypeNames.Asset,
        )
        nested_empty_texture.Set(Sdf.AssetPath(""))
        nested_empty_texture.GetAttr().SetColorSpace("sRGB")
        connected_source = material.CreateInput(
            "connected_texture_file",
            Sdf.ValueTypeNames.Asset,
        )
        connected_source.Set(Sdf.AssetPath("textures/connected.png"))
        connected_texture = shader.CreateInput(
            "connected_file", Sdf.ValueTypeNames.Asset
        )
        connected_texture.Set(Sdf.AssetPath(""))
        connected_texture.GetAttr().SetColorSpace("sRGB")
        connected_texture.ConnectToSource(connected_source)
        lib_stage.GetRootLayer().Save()

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        task._create_full_stage(
            input_usd_path=input_usd,
            output_usd_path=output_usd,
            resolved_materials={"Steel": "/World/Looks/Steel"},
            prim_to_material={"/World/Body": "Steel"},
            is_library_based=True,
            material_library_path=str(library_usd),
            flatten_output=True,
        )

        out_stage = Usd.Stage.Open(str(output_usd))
        out_material = out_stage.GetPrimAtPath("/World/Looks/Steel")
        out_empty_texture = out_material.GetAttribute("inputs:base_color_texture_file")
        out_real_texture = out_material.GetAttribute(
            "inputs:geometry_normal_texture_file"
        )
        out_shader = out_stage.GetPrimAtPath("/World/Looks/Steel/Shader")
        out_nested_empty_texture = out_shader.GetAttribute(
            "inputs:emissive_texture_file"
        )
        out_connected_texture = out_shader.GetAttribute("inputs:connected_file")

        assert out_empty_texture.Get() == Sdf.AssetPath("")
        assert not out_empty_texture.HasColorSpace()
        assert out_real_texture.Get() == Sdf.AssetPath("textures/normal.png")
        assert out_real_texture.HasColorSpace()
        assert out_real_texture.GetColorSpace() == "raw"
        assert out_nested_empty_texture.Get() == Sdf.AssetPath("")
        assert not out_nested_empty_texture.HasColorSpace()
        assert out_connected_texture.Get() == Sdf.AssetPath("")
        assert out_connected_texture.HasColorSpace()
        assert out_connected_texture.GetColorSpace() == "sRGB"
        assert out_connected_texture.GetConnections()

    def test_flatten_removes_unbound_input_mdl_shader_with_unresolved_mdl(
        self, tmp_path
    ):
        """Flattened output should not keep stale unresolved input MDL shaders."""
        input_usd = tmp_path / "input.usda"
        output_usd = tmp_path / "output.usd"
        library_usd = tmp_path / "library.usd"

        stage = Usd.Stage.CreateNew(str(input_usd))
        root = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(root.GetPrim())
        mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
        UsdGeom.Scope.Define(stage, "/World/Looks")
        default_mat = UsdShade.Material.Define(stage, "/World/Looks/DefaultMaterial")
        shader = UsdShade.Shader.Define(
            stage, "/World/Looks/DefaultMaterial/DefaultMaterial"
        )
        shader_prim = shader.GetPrim()
        shader_prim.CreateAttribute(
            "info:mdl:sourceAsset",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath("OmniPBR.mdl"))
        shader_prim.CreateAttribute(
            "info:mdl:sourceAsset:subIdentifier",
            Sdf.ValueTypeNames.Token,
        ).Set("OmniPBR")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(default_mat)
        stage.GetRootLayer().Save()

        lib_stage = Usd.Stage.CreateNew(str(library_usd))
        UsdGeom.Scope.Define(lib_stage, "/World")
        UsdGeom.Scope.Define(lib_stage, "/World/Looks")
        UsdShade.Material.Define(lib_stage, "/World/Looks/TestMaterial")
        lib_stage.GetRootLayer().Save()

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        task._create_full_stage(
            input_usd_path=input_usd,
            output_usd_path=output_usd,
            resolved_materials={"TestMaterial": "/World/Looks/TestMaterial"},
            prim_to_material={"/World/Mesh": "TestMaterial"},
            is_library_based=True,
            material_library_path=str(library_usd),
            flatten_output=True,
        )

        out_stage = Usd.Stage.Open(str(output_usd))

        stale_shader = out_stage.GetPrimAtPath(
            "/World/Looks/DefaultMaterial/DefaultMaterial"
        )
        assert not stale_shader or not stale_shader.IsActive()
        material, _relationship = UsdShade.MaterialBindingAPI(
            out_stage.GetPrimAtPath("/World/Mesh")
        ).ComputeBoundMaterial()
        assert material
        assert str(material.GetPath()) == "/World/Looks/TestMaterial"

        mdl_paths = []
        for prim in out_stage.Traverse():
            attr = prim.GetAttribute("info:mdl:sourceAsset")
            if attr and attr.IsValid() and attr.Get():
                mdl_paths.append(attr.Get().path)
        assert "OmniPBR.mdl" not in mdl_paths

    def test_flatten_removes_instance_prototype_unresolved_mdl_shader(self, tmp_path):
        """Cleanup should also sanitize prototype contents materialized by flatten."""
        input_usd = tmp_path / "input.usda"
        external_usd = tmp_path / "external.usda"
        output_usd = tmp_path / "output.usd"

        external_stage = Usd.Stage.CreateNew(str(external_usd))
        external_root = UsdGeom.Xform.Define(external_stage, "/Asset")
        external_stage.SetDefaultPrim(external_root.GetPrim())
        UsdGeom.Xform.Define(external_stage, "/Asset/Prototype")
        UsdGeom.Mesh.Define(external_stage, "/Asset/Prototype/Mesh")
        UsdShade.Material.Define(
            external_stage,
            "/Asset/Prototype/Looks/StaleMaterial",
        )
        stale_shader = UsdShade.Shader.Define(
            external_stage,
            "/Asset/Prototype/Looks/StaleMaterial/MDLShader",
        )
        stale_shader.GetPrim().CreateAttribute(
            "info:mdl:sourceAsset",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath("OmniPBR.mdl"))
        external_stage.GetRootLayer().Save()

        input_stage = Usd.Stage.CreateNew(str(input_usd))
        root = UsdGeom.Xform.Define(input_stage, "/World")
        input_stage.SetDefaultPrim(root.GetPrim())
        for name in ("InstA", "InstB"):
            instance = UsdGeom.Xform.Define(input_stage, f"/World/{name}")
            instance.GetPrim().GetReferences().AddReference(
                str(external_usd),
                "/Asset/Prototype",
            )
            instance.GetPrim().SetInstanceable(True)
        input_stage.GetRootLayer().Save()

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        task._create_full_stage(
            input_usd_path=input_usd,
            output_usd_path=output_usd,
            resolved_materials={},
            prim_to_material={},
            flatten_output=True,
        )

        output_layer = Sdf.Layer.FindOrOpen(str(output_usd))
        assert output_layer
        assert "OmniPBR.mdl" not in output_layer.ExportToString()

    def test_bound_input_material_with_unresolved_mdl_stays_active(self, tmp_path):
        """Cleanup must not deactivate materials that are still bound."""
        input_usd = tmp_path / "input.usda"

        stage = Usd.Stage.CreateNew(str(input_usd))
        root = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(root.GetPrim())
        mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
        material = UsdShade.Material.Define(stage, "/World/Looks/DefaultMaterial")
        shader = UsdShade.Shader.Define(
            stage, "/World/Looks/DefaultMaterial/DefaultMaterial"
        )
        shader.GetPrim().CreateAttribute(
            "info:mdl:sourceAsset",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath("OmniPBR.mdl"))
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
        stage.GetRootLayer().Save()

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        assert task._deactivate_unbound_unresolved_mdl_shaders(stage) == []
        assert stage.GetPrimAtPath("/World/Looks/DefaultMaterial").IsActive()
        assert stage.GetPrimAtPath(
            "/World/Looks/DefaultMaterial/DefaultMaterial"
        ).IsActive()

    def test_preview_bound_material_with_unresolved_mdl_stays_active(self, tmp_path):
        """Purpose-specific material bindings should protect material roots."""
        input_usd = tmp_path / "input.usda"

        stage = Usd.Stage.CreateNew(str(input_usd))
        root = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(root.GetPrim())
        mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
        material = UsdShade.Material.Define(stage, "/World/Looks/PreviewMaterial")
        shader = UsdShade.Shader.Define(stage, "/World/Looks/PreviewMaterial/MDLShader")
        shader.GetPrim().CreateAttribute(
            "info:mdl:sourceAsset",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath("OmniPBR.mdl"))
        shader_output = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput().ConnectToSource(shader_output)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
            material,
            UsdShade.Tokens.weakerThanDescendants,
            UsdShade.Tokens.preview,
        )
        stage.GetRootLayer().Save()

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        assert task._collect_bound_material_paths(stage) == {
            "/World/Looks/PreviewMaterial"
        }
        assert task._deactivate_unbound_unresolved_mdl_shaders(stage) == []
        assert shader.GetPrim().IsActive()

    def test_cyclic_material_graph_cleanup_terminates(self, tmp_path):
        """Cyclic shader connections should not hang stale shader cleanup."""
        input_usd = tmp_path / "input.usda"

        stage = Usd.Stage.CreateNew(str(input_usd))
        root = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(root.GetPrim())
        mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
        material = UsdShade.Material.Define(stage, "/World/Looks/CyclicMaterial")
        shader_a = UsdShade.Shader.Define(stage, "/World/Looks/CyclicMaterial/ShaderA")
        shader_b = UsdShade.Shader.Define(stage, "/World/Looks/CyclicMaterial/ShaderB")
        for shader in (shader_a, shader_b):
            shader.GetPrim().CreateAttribute(
                "info:mdl:sourceAsset",
                Sdf.ValueTypeNames.Asset,
            ).Set(Sdf.AssetPath("OmniPBR.mdl"))
        shader_a_output = shader_a.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        shader_b_output = shader_b.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        shader_a.CreateInput("cycle", Sdf.ValueTypeNames.Token).ConnectToSource(
            shader_b_output
        )
        shader_b.CreateInput("cycle", Sdf.ValueTypeNames.Token).ConnectToSource(
            shader_a_output
        )
        material.CreateSurfaceOutput().ConnectToSource(shader_a_output)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
        stage.GetRootLayer().Save()

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        assert task._collect_material_graph_prim_paths(stage, material.GetPrim()) == {
            "/World/Looks/CyclicMaterial",
            "/World/Looks/CyclicMaterial/ShaderA",
            "/World/Looks/CyclicMaterial/ShaderB",
        }
        assert task._deactivate_unbound_unresolved_mdl_shaders(stage) == []
        assert shader_a.GetPrim().IsActive()
        assert shader_b.GetPrim().IsActive()

    def test_protected_replacement_material_prunes_stale_child_shader(self, tmp_path):
        """Same-path replacement should remove obsolete unresolved child shaders."""
        input_usd = tmp_path / "input.usda"

        stage = Usd.Stage.CreateNew(str(input_usd))
        root = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(root.GetPrim())
        mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
        material = UsdShade.Material.Define(stage, "/World/Looks/TestMaterial")
        old_shader = UsdShade.Shader.Define(
            stage, "/World/Looks/TestMaterial/OldShader"
        )
        old_shader.GetPrim().CreateAttribute(
            "info:mdl:sourceAsset",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath("OmniPBR.mdl"))
        new_shader = UsdShade.Shader.Define(
            stage, "/World/Looks/TestMaterial/NewShader"
        )
        new_shader_output = new_shader.CreateOutput(
            "surface",
            Sdf.ValueTypeNames.Token,
        )
        material.CreateSurfaceOutput().ConnectToSource(new_shader_output)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
        stage.GetRootLayer().Save()

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        assert task._deactivate_unbound_unresolved_mdl_shaders(
            stage,
            protected_material_paths={"/World/Looks/TestMaterial"},
        ) == ["/World/Looks/TestMaterial/OldShader"]
        assert not old_shader.GetPrim().IsActive()
        assert new_shader.GetPrim().IsActive()
        assert "OmniPBR.mdl" not in stage.Flatten().ExportToString()

    def test_protected_material_keeps_connected_unresolved_mdl_shader(self, tmp_path):
        """Protected cleanup should keep shaders reached from material outputs."""
        input_usd = tmp_path / "input.usda"

        stage = Usd.Stage.CreateNew(str(input_usd))
        root = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(root.GetPrim())
        mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
        material = UsdShade.Material.Define(stage, "/World/Looks/TestMaterial")
        shader = UsdShade.Shader.Define(stage, "/World/Looks/TestMaterial/MDLShader")
        shader.GetPrim().CreateAttribute(
            "info:mdl:sourceAsset",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath("OmniPBR.mdl"))
        shader_output = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput().ConnectToSource(shader_output)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
        stage.GetRootLayer().Save()

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        assert (
            task._deactivate_unbound_unresolved_mdl_shaders(
                stage,
                protected_material_paths={"/World/Looks/TestMaterial"},
            )
            == []
        )
        assert shader.GetPrim().IsActive()

    def test_protected_replacement_ignores_stale_child_shader_connections(
        self, tmp_path
    ):
        """Unused old shaders should not protect other stale materials."""
        input_usd = tmp_path / "input.usda"

        stage = Usd.Stage.CreateNew(str(input_usd))
        root = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(root.GetPrim())
        mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
        material = UsdShade.Material.Define(stage, "/World/Looks/TestMaterial")
        old_shader = UsdShade.Shader.Define(
            stage, "/World/Looks/TestMaterial/OldShader"
        )
        old_shader.GetPrim().CreateAttribute(
            "info:mdl:sourceAsset",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath("OmniPBR.mdl"))
        UsdShade.Material.Define(stage, "/World/Looks/HelperMaterial")
        helper_shader = UsdShade.Shader.Define(
            stage, "/World/Looks/HelperMaterial/MDLShader"
        )
        helper_shader.GetPrim().CreateAttribute(
            "info:mdl:sourceAsset",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath("OmniPBR.mdl"))
        helper_output = helper_shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        old_shader.CreateInput("unused", Sdf.ValueTypeNames.Token).ConnectToSource(
            helper_output
        )
        new_shader = UsdShade.Shader.Define(
            stage, "/World/Looks/TestMaterial/NewShader"
        )
        new_shader_output = new_shader.CreateOutput(
            "surface",
            Sdf.ValueTypeNames.Token,
        )
        material.CreateSurfaceOutput().ConnectToSource(new_shader_output)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
        stage.GetRootLayer().Save()

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        deactivated_paths = task._deactivate_unbound_unresolved_mdl_shaders(
            stage,
            protected_material_paths={"/World/Looks/TestMaterial"},
        )

        assert set(deactivated_paths) == {
            "/World/Looks/TestMaterial/OldShader",
            "/World/Looks/HelperMaterial/MDLShader",
        }
        assert not old_shader.GetPrim().IsActive()
        assert not helper_shader.GetPrim().IsActive()
        assert new_shader.GetPrim().IsActive()

    def test_remote_resolved_mdl_uri_is_not_treated_as_missing(self, tmp_path):
        """Remote resolved asset URIs should not go through local Path.exists()."""

        class AssetValue:
            path = "Materials/RemoteMaterial.mdl"
            resolvedPath = "omniverse://server.example/Materials/RemoteMaterial.mdl"

        task = ApplyMaterialsToUSDTask()

        assert not task._is_uri_asset_path("C:/Materials/RemoteMaterial.mdl")
        assert not task._is_uri_asset_path(r"C:\Materials\RemoteMaterial.mdl")
        assert not task._is_unresolved_local_asset_path(
            AssetValue(),
            AssetValue.path,
            tmp_path,
        )

    def test_authored_mdl_uri_is_treated_as_unsafe(self, tmp_path):
        """Authored resolver URIs must not survive into generated USD output."""

        class AssetValue:
            path = "https://metadata.example.invalid/Materials/Evil.mdl"
            resolvedPath = ""

        task = ApplyMaterialsToUSDTask()

        assert task._is_uri_asset_path(AssetValue.path)
        assert task._is_unresolved_local_asset_path(
            AssetValue(),
            AssetValue.path,
            tmp_path,
        )

    def test_make_path_relative_rejects_resolver_uri_material(self, tmp_path):
        """LLM-provided material URLs should fail before OVRTX can resolve them."""
        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        with pytest.raises(ValueError, match="resolver URI material path"):
            task._make_path_relative_to_usd(
                "https://metadata.example.invalid/Evil.mdl",
                tmp_path / "output.usda",
            )

        with pytest.raises(ValueError, match="resolver URI material path"):
            task._make_path_relative_to_usd(
                "file:///var/run/secrets/kubernetes.io/serviceaccount/token",
                tmp_path / "output.usda",
            )

    def test_remap_single_asset_path_clears_unsafe_resolver_paths(self, tmp_path):
        """Copied material libraries should not author URI or host-absolute assets."""
        source_dir = tmp_path / "library"
        target_dir = tmp_path / "out"
        source_dir.mkdir()
        target_dir.mkdir()
        local_asset = source_dir / "textures" / "albedo.png"
        local_asset.parent.mkdir()
        local_asset.write_bytes(b"png")

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        assert (
            task._remap_single_asset_path(
                str(local_asset),
                source_dir,
                target_dir,
            )
            == "../library/textures/albedo.png"
        )
        for unsafe_path in (
            "https://metadata.example.invalid/albedo.png",
            "file:///etc/shadow",
            "/etc/shadow",
            "C:/Users/secret/material.mdl",
            "../outside/material.mdl",
        ):
            assert (
                task._remap_single_asset_path(
                    unsafe_path,
                    source_dir,
                    target_dir,
                )
                == ""
            )

    def test_windows_absolute_mdl_path_is_not_treated_as_uri(self, tmp_path):
        """Windows drive paths should still be checked as local asset paths."""

        class AssetValue:
            path = "C:/Materials/MissingMaterial.mdl"
            resolvedPath = ""

        task = ApplyMaterialsToUSDTask()

        assert task._is_absolute_asset_path(AssetValue.path)
        assert not task._is_uri_asset_path(AssetValue.path)
        assert task._is_unresolved_local_asset_path(
            AssetValue(),
            AssetValue.path,
            tmp_path,
        )

    def test_resolved_mdl_package_path_is_not_treated_as_missing(self, tmp_path):
        """Any non-empty resolver path should be trusted, including package paths."""

        class AssetValue:
            path = "Materials/PackagedMaterial.mdl"
            resolvedPath = "asset.usdz[Materials/PackagedMaterial.mdl]"

        task = ApplyMaterialsToUSDTask()

        assert not task._is_unresolved_local_asset_path(
            AssetValue(),
            AssetValue.path,
            tmp_path,
        )

    def test_layer_relative_asset_fallback_checks_authored_layer_dir(self, tmp_path):
        """Fallback local checks should honor the layer that authored the asset."""

        class AssetValue:
            path = "materials/LayerMaterial.mdl"
            resolvedPath = ""

        subdir = tmp_path / "layers"
        material_dir = subdir / "materials"
        material_dir.mkdir(parents=True)
        (material_dir / "LayerMaterial.mdl").write_text("mdl", encoding="utf-8")

        sublayer_path = subdir / "materials.usda"
        sublayer_stage = Usd.Stage.CreateNew(str(sublayer_path))
        material = UsdShade.Material.Define(
            sublayer_stage,
            "/World/Looks/LayerMaterial",
        )
        shader = UsdShade.Shader.Define(
            sublayer_stage,
            "/World/Looks/LayerMaterial/MDLShader",
        )
        shader.GetPrim().CreateAttribute(
            "info:mdl:sourceAsset",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath(AssetValue.path))
        shader_output = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput().ConnectToSource(shader_output)
        sublayer_stage.GetRootLayer().Save()

        root_path = tmp_path / "root.usda"
        root_stage = Usd.Stage.CreateNew(str(root_path))
        root_stage.GetRootLayer().subLayerPaths.append(str(sublayer_path))
        root_stage.GetRootLayer().Save()
        stage = Usd.Stage.Open(str(root_path))
        attr = stage.GetPrimAtPath("/World/Looks/LayerMaterial/MDLShader").GetAttribute(
            "info:mdl:sourceAsset"
        )

        task = ApplyMaterialsToUSDTask()

        base_dirs = task._asset_base_dirs_for_attr(stage, attr)
        assert subdir in base_dirs
        assert not task._is_unresolved_local_asset_path(
            AssetValue(),
            AssetValue.path,
            base_dirs,
        )

    def test_unbound_unresolved_mdl_cleanup_preserves_fallback_shader(self, tmp_path):
        """Only stale unresolved MDL shaders should be hidden, not whole materials."""
        input_usd = tmp_path / "input.usda"

        stage = Usd.Stage.CreateNew(str(input_usd))
        root = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(root.GetPrim())
        material = UsdShade.Material.Define(stage, "/World/Looks/DefaultMaterial")
        mdl_shader = UsdShade.Shader.Define(
            stage, "/World/Looks/DefaultMaterial/MDLShader"
        )
        mdl_shader.GetPrim().CreateAttribute(
            "info:mdl:sourceAsset",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath("OmniPBR.mdl"))
        preview_shader = UsdShade.Shader.Define(
            stage, "/World/Looks/DefaultMaterial/PreviewShader"
        )
        stage.GetRootLayer().Save()

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        assert task._deactivate_unbound_unresolved_mdl_shaders(stage) == [
            "/World/Looks/DefaultMaterial/MDLShader"
        ]
        assert material.GetPrim().IsActive()
        assert not mdl_shader.GetPrim().IsActive()
        assert preview_shader.GetPrim().IsActive()

    def test_unbound_material_with_loose_shader_is_deactivated(self, tmp_path):
        """Stale material graphs can use sibling shaders outside the material."""
        input_usd = tmp_path / "input.usda"

        stage = Usd.Stage.CreateNew(str(input_usd))
        root = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(root.GetPrim())
        old_material = UsdShade.Material.Define(stage, "/World/Looks/OldMaterial")
        old_shader = UsdShade.Shader.Define(stage, "/World/Looks/OldMaterialShader")
        old_shader.GetPrim().CreateAttribute(
            "info:mdl:sourceAsset",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath("OmniPBR.mdl"))
        old_shader_output = old_shader.CreateOutput(
            "surface",
            Sdf.ValueTypeNames.Token,
        )
        old_material.CreateSurfaceOutput().ConnectToSource(old_shader_output)
        stage.GetRootLayer().Save()

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        assert task._deactivate_unbound_unresolved_mdl_shaders(stage) == [
            "/World/Looks/OldMaterialShader"
        ]
        assert not old_shader.GetPrim().IsActive()
        assert "OmniPBR.mdl" not in stage.Flatten().ExportToString()

    def test_bound_material_with_loose_shader_stays_active(self, tmp_path):
        """Loose shaders used by resolved bound materials remain protected."""
        input_usd = tmp_path / "input.usda"

        stage = Usd.Stage.CreateNew(str(input_usd))
        root = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(root.GetPrim())
        mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
        bound_material = UsdShade.Material.Define(stage, "/World/Looks/BoundMaterial")
        stale_material = UsdShade.Material.Define(stage, "/World/Looks/StaleMaterial")
        shared_shader = UsdShade.Shader.Define(stage, "/World/Looks/SharedShader")
        shared_shader.GetPrim().CreateAttribute(
            "info:mdl:sourceAsset",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath("OmniPBR.mdl"))
        shared_output = shared_shader.CreateOutput(
            "surface",
            Sdf.ValueTypeNames.Token,
        )
        bound_material.CreateSurfaceOutput().ConnectToSource(shared_output)
        stale_material.CreateSurfaceOutput().ConnectToSource(shared_output)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(bound_material)
        stage.GetRootLayer().Save()

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        assert task._deactivate_unbound_unresolved_mdl_shaders(stage) == []
        assert shared_shader.GetPrim().IsActive()

    def test_standalone_loose_unresolved_mdl_shader_is_deactivated(self, tmp_path):
        """Loose unresolved shaders outside any reachable graph should be hidden."""
        input_usd = tmp_path / "input.usda"

        stage = Usd.Stage.CreateNew(str(input_usd))
        root = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(root.GetPrim())
        loose_shader = UsdShade.Shader.Define(stage, "/World/Looks/LooseShader")
        loose_shader.GetPrim().CreateAttribute(
            "info:mdl:sourceAsset",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath("OmniPBR.mdl"))
        stage.GetRootLayer().Save()

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        assert task._deactivate_unbound_unresolved_mdl_shaders(stage) == [
            "/World/Looks/LooseShader"
        ]
        assert not loose_shader.GetPrim().IsActive()
        assert "OmniPBR.mdl" not in stage.Flatten().ExportToString()

    def test_composition_target_material_with_unresolved_mdl_is_protected(
        self, tmp_path
    ):
        """Materials used as inherit bases should not be treated as stale."""
        input_usd = tmp_path / "input.usda"

        stage = Usd.Stage.CreateNew(str(input_usd))
        root = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(root.GetPrim())
        mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
        base_material = UsdShade.Material.Define(stage, "/World/Looks/BaseMaterial")
        base_shader = UsdShade.Shader.Define(
            stage, "/World/Looks/BaseMaterial/MDLShader"
        )
        base_shader.GetPrim().CreateAttribute(
            "info:mdl:sourceAsset",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath("OmniPBR.mdl"))
        child_material = UsdShade.Material.Define(stage, "/World/Looks/ChildMaterial")
        child_material.GetPrim().GetInherits().AddInherit("/World/Looks/BaseMaterial")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(child_material)
        stage.GetRootLayer().Save()

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        assert task._deactivate_unbound_unresolved_mdl_shaders(stage) == []
        assert base_material.GetPrim().IsActive()
        assert base_shader.GetPrim().IsActive()

    def test_specializes_target_material_with_unresolved_mdl_is_protected(
        self, tmp_path
    ):
        """Materials used as specializes bases should not be treated as stale."""
        input_usd = tmp_path / "input.usda"

        stage = Usd.Stage.CreateNew(str(input_usd))
        root = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(root.GetPrim())
        mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
        base_material = UsdShade.Material.Define(stage, "/World/Looks/BaseMaterial")
        base_shader = UsdShade.Shader.Define(
            stage, "/World/Looks/BaseMaterial/MDLShader"
        )
        base_shader.GetPrim().CreateAttribute(
            "info:mdl:sourceAsset",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath("OmniPBR.mdl"))
        child_material = UsdShade.Material.Define(stage, "/World/Looks/ChildMaterial")
        child_material.GetPrim().GetSpecializes().AddSpecialize(
            "/World/Looks/BaseMaterial"
        )
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(child_material)
        stage.GetRootLayer().Save()

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        assert task._deactivate_unbound_unresolved_mdl_shaders(stage) == []
        assert base_material.GetPrim().IsActive()
        assert base_shader.GetPrim().IsActive()

    def test_payload_target_material_with_unresolved_mdl_is_protected(self, tmp_path):
        """Materials used as payload bases should not be treated as stale."""
        input_usd = tmp_path / "input.usda"

        stage = Usd.Stage.CreateNew(str(input_usd))
        root = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(root.GetPrim())
        mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
        base_material = UsdShade.Material.Define(stage, "/World/Looks/BaseMaterial")
        base_shader = UsdShade.Shader.Define(
            stage, "/World/Looks/BaseMaterial/MDLShader"
        )
        base_shader.GetPrim().CreateAttribute(
            "info:mdl:sourceAsset",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath("OmniPBR.mdl"))
        child_material = UsdShade.Material.Define(stage, "/World/Looks/ChildMaterial")
        child_material.GetPrim().GetPayloads().AddInternalPayload(
            "/World/Looks/BaseMaterial"
        )
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(child_material)
        stage.GetRootLayer().Save()

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        assert task._deactivate_unbound_unresolved_mdl_shaders(stage) == []
        assert base_material.GetPrim().IsActive()
        assert base_shader.GetPrim().IsActive()

    def test_shader_composition_target_material_with_unresolved_mdl_is_protected(
        self, tmp_path
    ):
        """Materials owning shader composition targets should not be stale."""
        input_usd = tmp_path / "input.usda"

        stage = Usd.Stage.CreateNew(str(input_usd))
        root = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(root.GetPrim())
        mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
        base_material = UsdShade.Material.Define(stage, "/World/Looks/BaseMaterial")
        base_shader = UsdShade.Shader.Define(
            stage, "/World/Looks/BaseMaterial/MDLShader"
        )
        base_shader.GetPrim().CreateAttribute(
            "info:mdl:sourceAsset",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath("OmniPBR.mdl"))
        child_material = UsdShade.Material.Define(stage, "/World/Looks/ChildMaterial")
        child_shader = UsdShade.Shader.Define(
            stage, "/World/Looks/ChildMaterial/MDLShader"
        )
        child_shader.GetPrim().GetInherits().AddInherit(
            "/World/Looks/BaseMaterial/MDLShader"
        )
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(child_material)
        stage.GetRootLayer().Save()

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        assert task._deactivate_unbound_unresolved_mdl_shaders(stage) == []
        assert base_material.GetPrim().IsActive()
        assert base_shader.GetPrim().IsActive()

    def test_connected_shader_material_with_unresolved_mdl_is_protected(self, tmp_path):
        """Materials reached through bound material connections should be active."""
        input_usd = tmp_path / "input.usda"

        stage = Usd.Stage.CreateNew(str(input_usd))
        root = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(root.GetPrim())
        mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
        bound_material = UsdShade.Material.Define(stage, "/World/Looks/BoundMaterial")
        helper_material = UsdShade.Material.Define(stage, "/World/Looks/HelperMaterial")
        helper_shader = UsdShade.Shader.Define(
            stage, "/World/Looks/HelperMaterial/MDLShader"
        )
        helper_shader.GetPrim().CreateAttribute(
            "info:mdl:sourceAsset",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath("OmniPBR.mdl"))
        helper_output = helper_shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        bound_material.CreateSurfaceOutput().ConnectToSource(helper_output)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(bound_material)
        stage.GetRootLayer().Save()

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        assert task._deactivate_unbound_unresolved_mdl_shaders(stage) == []
        assert helper_material.GetPrim().IsActive()
        assert helper_shader.GetPrim().IsActive()

    def test_collection_binding_does_not_protect_unrelated_stale_material(
        self, tmp_path
    ):
        """Collection targets should not become material roots for reachability."""
        input_usd = tmp_path / "input.usda"

        stage = Usd.Stage.CreateNew(str(input_usd))
        root = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(root.GetPrim())
        mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
        bound_material = UsdShade.Material.Define(stage, "/World/Looks/BoundMaterial")
        bound_shader = UsdShade.Shader.Define(
            stage, "/World/Looks/BoundMaterial/MDLShader"
        )
        bound_shader.GetPrim().CreateAttribute(
            "info:mdl:sourceAsset",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath("OmniPBR.mdl"))
        bound_output = bound_shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        bound_material.CreateSurfaceOutput().ConnectToSource(bound_output)
        stale_material = UsdShade.Material.Define(stage, "/World/Looks/StaleMaterial")
        stale_shader = UsdShade.Shader.Define(
            stage, "/World/Looks/StaleMaterial/MDLShader"
        )
        stale_shader.GetPrim().CreateAttribute(
            "info:mdl:sourceAsset",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath("OmniPBR.mdl"))
        stale_output = stale_shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        stale_material.CreateSurfaceOutput().ConnectToSource(stale_output)

        collection = Usd.CollectionAPI.Apply(root.GetPrim(), "all")
        collection.GetIncludesRel().AddTarget(mesh.GetPath())
        UsdShade.MaterialBindingAPI.Apply(root.GetPrim()).Bind(
            collection,
            bound_material,
        )
        stage.GetRootLayer().Save()

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        assert task._deactivate_unbound_unresolved_mdl_shaders(stage) == [
            "/World/Looks/StaleMaterial/MDLShader"
        ]
        assert bound_shader.GetPrim().IsActive()
        assert not stale_shader.GetPrim().IsActive()

    def test_overridden_collection_binding_does_not_protect_stale_material(
        self, tmp_path
    ):
        """Only resolved material bindings should protect material roots."""
        input_usd = tmp_path / "input.usda"

        stage = Usd.Stage.CreateNew(str(input_usd))
        root = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(root.GetPrim())
        mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
        stale_material = UsdShade.Material.Define(stage, "/World/Looks/StaleMaterial")
        stale_shader = UsdShade.Shader.Define(
            stage, "/World/Looks/StaleMaterial/MDLShader"
        )
        stale_shader.GetPrim().CreateAttribute(
            "info:mdl:sourceAsset",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath("OmniPBR.mdl"))
        stale_output = stale_shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        stale_material.CreateSurfaceOutput().ConnectToSource(stale_output)
        bound_material = UsdShade.Material.Define(stage, "/World/Looks/BoundMaterial")
        bound_shader = UsdShade.Shader.Define(
            stage, "/World/Looks/BoundMaterial/MDLShader"
        )
        bound_shader.GetPrim().CreateAttribute(
            "info:mdl:sourceAsset",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath("OmniPBR.mdl"))
        bound_output = bound_shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        bound_material.CreateSurfaceOutput().ConnectToSource(bound_output)

        collection = Usd.CollectionAPI.Apply(root.GetPrim(), "all")
        collection.GetIncludesRel().AddTarget(mesh.GetPath())
        UsdShade.MaterialBindingAPI.Apply(root.GetPrim()).Bind(
            collection,
            stale_material,
        )
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(bound_material)
        stage.GetRootLayer().Save()

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        assert task._collect_bound_material_paths(stage) == {
            "/World/Looks/BoundMaterial"
        }
        assert task._deactivate_unbound_unresolved_mdl_shaders(stage) == [
            "/World/Looks/StaleMaterial/MDLShader"
        ]
        assert bound_shader.GetPrim().IsActive()
        assert not stale_shader.GetPrim().IsActive()

    def test_empty_material_binding_relationship_is_ignored(self, tmp_path):
        """Cleared bindings should not be resolved as material targets."""
        input_usd = tmp_path / "input.usda"

        stage = Usd.Stage.CreateNew(str(input_usd))
        root = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(root.GetPrim())
        mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
        mesh.GetPrim().CreateRelationship("material:binding").SetTargets([])
        UsdShade.Material.Define(stage, "/World/Looks/StaleMaterial")
        stale_shader = UsdShade.Shader.Define(
            stage, "/World/Looks/StaleMaterial/MDLShader"
        )
        stale_shader.GetPrim().CreateAttribute(
            "info:mdl:sourceAsset",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath("OmniPBR.mdl"))
        stage.GetRootLayer().Save()

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        assert task._collect_bound_material_paths(stage) == set()
        assert task._deactivate_unbound_unresolved_mdl_shaders(stage) == [
            "/World/Looks/StaleMaterial/MDLShader"
        ]
        assert not stale_shader.GetPrim().IsActive()

    def test_unbound_class_material_shader_with_unresolved_mdl_is_deactivated(
        self, tmp_path
    ):
        """Cleanup should include abstract class material templates."""
        input_usd = tmp_path / "input.usda"

        stage = Usd.Stage.CreateNew(str(input_usd))
        root = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(root.GetPrim())
        class_material = UsdShade.Material.Define(stage, "/World/Looks/ClassMaterial")
        class_material.GetPrim().SetSpecifier(Sdf.SpecifierClass)
        class_shader = UsdShade.Shader.Define(
            stage, "/World/Looks/ClassMaterial/MDLShader"
        )
        class_shader.GetPrim().SetSpecifier(Sdf.SpecifierClass)
        class_shader.GetPrim().CreateAttribute(
            "info:mdl:sourceAsset",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath("OmniPBR.mdl"))
        stage.GetRootLayer().Save()

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        assert class_material.GetPrim().IsAbstract()
        assert class_shader.GetPrim().IsAbstract()
        assert task._deactivate_unbound_unresolved_mdl_shaders(stage) == [
            "/World/Looks/ClassMaterial/MDLShader"
        ]
        assert not class_shader.GetPrim().IsActive()
        assert "OmniPBR.mdl" not in stage.Flatten().ExportToString()

    def test_stale_composition_base_with_unresolved_mdl_is_not_protected(
        self, tmp_path
    ):
        """Unbound material composition graphs should not protect stale bases."""
        input_usd = tmp_path / "input.usda"

        stage = Usd.Stage.CreateNew(str(input_usd))
        root = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(root.GetPrim())
        mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
        bound_material = UsdShade.Material.Define(stage, "/World/Looks/BoundMaterial")
        UsdShade.Material.Define(stage, "/World/Looks/BaseMaterial")
        base_shader = UsdShade.Shader.Define(
            stage, "/World/Looks/BaseMaterial/MDLShader"
        )
        base_shader.GetPrim().CreateAttribute(
            "info:mdl:sourceAsset",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath("OmniPBR.mdl"))
        stale_child = UsdShade.Material.Define(stage, "/World/Looks/StaleChild")
        stale_child.GetPrim().GetInherits().AddInherit("/World/Looks/BaseMaterial")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(bound_material)
        stage.GetRootLayer().Save()

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        assert task._deactivate_unbound_unresolved_mdl_shaders(stage) == [
            "/World/Looks/BaseMaterial/MDLShader"
        ]
        assert not base_shader.GetPrim().IsActive()

    def test_external_reference_prim_path_does_not_protect_local_stale_material(
        self, tmp_path
    ):
        """External reference prim paths should not resolve in the current stage."""
        input_usd = tmp_path / "input.usda"
        external_usd = tmp_path / "external.usda"

        external_stage = Usd.Stage.CreateNew(str(external_usd))
        external_root = UsdGeom.Xform.Define(external_stage, "/World")
        external_stage.SetDefaultPrim(external_root.GetPrim())
        UsdShade.Material.Define(external_stage, "/World/Looks/BaseMaterial")
        external_stage.GetRootLayer().Save()

        stage = Usd.Stage.CreateNew(str(input_usd))
        root = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(root.GetPrim())
        mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
        bound_material = UsdShade.Material.Define(stage, "/World/Looks/BoundMaterial")
        bound_material.GetPrim().GetReferences().AddReference(
            str(external_usd),
            "/World/Looks/BaseMaterial",
        )
        UsdShade.Material.Define(stage, "/World/Looks/BaseMaterial")
        stale_shader = UsdShade.Shader.Define(
            stage, "/World/Looks/BaseMaterial/MDLShader"
        )
        stale_shader.GetPrim().CreateAttribute(
            "info:mdl:sourceAsset",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath("OmniPBR.mdl"))
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(bound_material)
        stage.GetRootLayer().Save()

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        assert task._deactivate_unbound_unresolved_mdl_shaders(stage) == [
            "/World/Looks/BaseMaterial/MDLShader"
        ]
        assert not stale_shader.GetPrim().IsActive()

    def test_nested_bound_material_shader_is_not_deactivated_by_stale_parent(
        self, tmp_path
    ):
        """Cleanup should not cross nested material ownership boundaries."""
        input_usd = tmp_path / "input.usda"

        stage = Usd.Stage.CreateNew(str(input_usd))
        root = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(root.GetPrim())
        mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
        parent_material = UsdShade.Material.Define(stage, "/World/Looks/ParentMaterial")
        parent_shader = UsdShade.Shader.Define(
            stage, "/World/Looks/ParentMaterial/MDLShader"
        )
        parent_shader.GetPrim().CreateAttribute(
            "info:mdl:sourceAsset",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath("OmniPBR.mdl"))
        child_material = UsdShade.Material.Define(
            stage, "/World/Looks/ParentMaterial/ChildMaterial"
        )
        child_shader = UsdShade.Shader.Define(
            stage, "/World/Looks/ParentMaterial/ChildMaterial/MDLShader"
        )
        child_shader.GetPrim().CreateAttribute(
            "info:mdl:sourceAsset",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath("OmniPBR.mdl"))
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(child_material)
        stage.GetRootLayer().Save()

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        assert task._deactivate_unbound_unresolved_mdl_shaders(stage) == [
            "/World/Looks/ParentMaterial/MDLShader"
        ]
        assert parent_material.GetPrim().IsActive()
        assert not parent_shader.GetPrim().IsActive()
        assert child_material.GetPrim().IsActive()
        assert child_shader.GetPrim().IsActive()


class TestApplyMaterialsHelperCoverage:
    def test_asset_path_remap_and_color_space_helpers(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        source_dir = tmp_path / "library"
        target_dir = tmp_path / "output"
        (source_dir / "textures").mkdir(parents=True)
        target_dir.mkdir()
        (source_dir / "textures" / "a.png").write_bytes(b"png")
        (source_dir / "textures" / "b.png").write_bytes(b"png")

        assert (
            apply_module.remap_single_asset_path(
                "https://example.invalid/tex.png",
                source_dir,
                target_dir,
            )
            == ""
        )
        with monkeypatch.context() as scoped:
            scoped.setattr(
                apply_module.os.path,
                "relpath",
                MagicMock(side_effect=ValueError("different drives")),
            )
            assert (
                apply_module.remap_single_asset_path(
                    "textures/a.png",
                    source_dir,
                    target_dir,
                )
                == ""
            )

        layer = Sdf.Layer.CreateAnonymous("materials.usda")
        assert (
            apply_module.clear_color_space_on_empty_asset_inputs(
                layer,
                Sdf.Path("/Missing"),
            )
            == 0
        )
        apply_module.remap_asset_paths_in_prim(
            layer,
            Sdf.Path("/Missing"),
            source_dir,
            target_dir,
        )

        prim_spec = Sdf.CreatePrimInLayer(layer, "/Looks/Mat")
        prim_spec.specifier = Sdf.SpecifierDef
        asset_attr = Sdf.AttributeSpec(
            prim_spec,
            "inputs:file",
            Sdf.ValueTypeNames.Asset,
        )
        asset_attr.default = Sdf.AssetPath("textures/a.png")
        empty_asset_attr = Sdf.AttributeSpec(
            prim_spec,
            "inputs:empty",
            Sdf.ValueTypeNames.Asset,
        )
        empty_asset_attr.default = Sdf.AssetPath("")
        empty_asset_attr.SetInfo("colorSpace", "sRGB")
        array_attr = Sdf.AttributeSpec(
            prim_spec,
            "inputs:files",
            Sdf.ValueTypeNames.AssetArray,
        )
        array_attr.default = Sdf.AssetPathArray(
            [
                Sdf.AssetPath("textures/a.png"),
                Sdf.AssetPath("textures/b.png"),
            ]
        )

        apply_module.remap_asset_paths_in_prim(
            layer,
            Sdf.Path("/Looks/Mat"),
            source_dir,
            target_dir,
        )
        cleared = apply_module.clear_color_space_on_empty_asset_inputs(
            layer,
            Sdf.Path("/Looks/Mat"),
        )

        assert asset_attr.default.path == "../library/textures/a.png"
        assert [asset.path for asset in array_attr.default] == [
            "../library/textures/a.png",
            "../library/textures/b.png",
        ]
        assert cleared == 1
        assert not empty_asset_attr.HasInfo("colorSpace")

    def test_shader_profile_helpers_cover_error_and_preview_paths(
        self, tmp_path: Path
    ) -> None:
        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        class RaisingOutput:
            def GetConnectedSources(self):
                raise RuntimeError("bad connection")

        class EmptySource:
            def GetPrim(self):
                return Usd.Prim()

        class FakeOutput:
            def GetConnectedSources(self):
                return (
                    [
                        type("SourceInfo", (), {"source": None})(),
                        type("SourceInfo", (), {"source": EmptySource()})(),
                    ],
                    None,
                )

        assert task._connected_shader_ids(RaisingOutput()) == set()
        assert task._connected_shader_ids(FakeOutput()) == set()

        stage_path = tmp_path / "profiles.usda"
        stage = Usd.Stage.CreateNew(str(stage_path))
        assert task._detect_material_profile_on_stage(stage, "/Missing")[0] is None

        material = UsdShade.Material.Define(stage, "/World/Looks/Preview")
        shader = UsdShade.Shader.Define(stage, "/World/Looks/Preview/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader_output = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput().ConnectToSource(shader_output)

        profile, details = task._detect_material_profile_on_stage(
            stage,
            "/World/Looks/Preview",
        )

        assert profile == "preview_surface"
        assert details["has_preview_surface"] is True

    def test_profile_resolution_edge_branches(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        monkeypatch.setattr(
            task,
            "_source_profile_for_file_material",
            lambda material_name, _material_path: {
                "UnknownGraph": (
                    None,
                    None,
                    {
                        "has_openpbr_materialx": False,
                        "has_preview_surface": False,
                        "has_mdl": False,
                    },
                ),
                "Preview": (
                    "preview_surface",
                    None,
                    {
                        "has_openpbr_materialx": False,
                        "has_preview_surface": True,
                        "has_mdl": False,
                    },
                ),
                "MDL": (
                    "omnipbr_mdl",
                    None,
                    {
                        "has_openpbr_materialx": False,
                        "has_preview_surface": False,
                        "has_mdl": True,
                    },
                ),
            }[material_name],
        )

        auto_unknown = task._resolve_material_profile_request(
            requested_profile="auto",
            resolved_materials={"UnknownGraph": "unknown.usd"},
            is_library_based=False,
            material_library_path=None,
        )
        preview_ok = task._resolve_material_profile_request(
            requested_profile="preview_surface",
            resolved_materials={"Preview": "preview.usd"},
            is_library_based=False,
            material_library_path=None,
        )
        preview_error = task._resolve_material_profile_request(
            requested_profile="preview_surface",
            resolved_materials={"UnknownGraph": "unknown.usd"},
            is_library_based=False,
            material_library_path=None,
        )
        mixed = task._resolve_material_profile_request(
            requested_profile="auto",
            resolved_materials={"Preview": "preview.usd", "MDL": "mat.mdl"},
            is_library_based=False,
            material_library_path=None,
        )
        explicit_mdl = task._resolve_material_profile_request(
            requested_profile="omnipbr_mdl",
            resolved_materials={"MDL": "mat.mdl"},
            is_library_based=False,
            material_library_path=None,
        )

        assert auto_unknown["fallback_reason"] == (
            "unrecognized_material_graph_uses_preview_surface"
        )
        assert (
            preview_ok["materials"]["Preview"]["resolved_profile"] == "preview_surface"
        )
        assert preview_error["errors"][0]["code"] == (
            "material_profile.unsupported_preview_surface_source"
        )
        assert mixed["resolved_profile"] == "mixed"
        assert explicit_mdl["materials"]["MDL"]["resolved_profile"] == "omnipbr_mdl"

        monkeypatch.undo()
        fallback_result = task._resolve_material_profile_request(
            requested_profile="openpbr_materialx",
            resolved_materials={FALLBACK_MATERIAL_NAME: "fallback.usd"},
            is_library_based=False,
            material_library_path=None,
        )
        assert fallback_result["fallback_reason"] == "canonical_fallback_material"
        assert (
            task._authoring_backend_for_material_profile(
                requested_profile="auto",
                resolved_profile="openpbr_materialx",
                source_profile="openpbr_materialx",
                fallback_reason=None,
                is_library_based=True,
            )
            == "preserved_library_material_graph"
        )
        assert (
            task._authoring_backend_for_material_profile(
                requested_profile="auto",
                resolved_profile="preview_surface",
                source_profile=None,
                fallback_reason="canonical_fallback_material",
                is_library_based=True,
            )
            == "usdshade.UsdPreviewSurface"
        )
        assert (
            task._authoring_backend_for_material_profile(
                requested_profile="auto",
                resolved_profile="custom_graph",
                source_profile=None,
                fallback_reason=None,
                is_library_based=False,
            )
            == "preserved_or_existing_material_graph"
        )
        assert (
            task._authoring_backend_for_material_profile(
                requested_profile="auto",
                resolved_profile=None,
                source_profile=None,
                fallback_reason=None,
                is_library_based=False,
            )
            == "unresolved"
        )
        assert task._fallback_material_path("", "Fallback") == "/Materials/Fallback"
        assert task._fallback_material_path("/World", "Fallback") == (
            "/World/Looks/Fallback"
        )

    def test_material_has_mdl_and_binding_schema_edges(self, tmp_path: Path) -> None:
        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()
        stage = Usd.Stage.CreateNew(str(tmp_path / "stage.usda"))
        material = UsdShade.Material.Define(stage, "/World/Looks/MDL")
        shader = UsdShade.Shader.Define(stage, "/World/Looks/MDL/Shader")
        shader.GetPrim().CreateAttribute(
            "info:mdl:sourceAsset",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath("OmniPBR.mdl"))
        empty_material = UsdShade.Material.Define(stage, "/World/Looks/Empty")

        assert task._material_has_mdl_shader(material.GetPrim()) is True
        assert (
            task._detect_material_profile_on_stage(
                stage, str(empty_material.GetPath())
            )[0]
            is None
        )

        layer = Sdf.Layer.CreateAnonymous("bindings.usda")
        task._author_material_binding_in_layer(layer, "/World/Mesh", "/World/Looks/MDL")
        mesh_spec = layer.GetPrimAtPath("/World/Mesh")
        assert "MaterialBindingAPI" in mesh_spec.GetInfo("apiSchemas").prependedItems

        explicit_spec = Sdf.CreatePrimInLayer(layer, "/World/ExplicitMesh")
        explicit_spec.specifier = Sdf.SpecifierOver
        explicit_api_schemas = Sdf.TokenListOp()
        explicit_api_schemas.explicitItems = ["ExistingAPI"]
        explicit_spec.SetInfo("apiSchemas", explicit_api_schemas)
        task._author_material_binding_in_layer(
            layer,
            "/World/ExplicitMesh",
            "/World/Looks/MDL",
        )
        assert "MaterialBindingAPI" in explicit_spec.GetInfo("apiSchemas").explicitItems

    def test_prediction_mapping_and_counting_edge_shapes(self, tmp_path: Path) -> None:
        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        assert task._load_prim_material_mapping(tmp_path / "missing.jsonl") == {}
        assert list(task._iter_prediction_mapping_records(123)) == []
        assert task._prediction_material_value({"materials": "Steel"}) == (
            True,
            "Steel",
        )
        assert task._prediction_material_value(
            {"validation_status": "disallowed_unknown"}
        ) == (True, "__UNKNOWN__")
        assert task._count_prediction_materials(None) == {
            "total": 0,
            "actionable": 0,
            "unknown": 0,
            "missing": 0,
        }
        assert list(task._iter_prediction_material_values(123)) == []

        predictions_path = tmp_path / "predictions.jsonl"
        predictions_path.write_text(
            "\n".join(
                [
                    "",
                    json.dumps({"id": "/A", "materials": "Steel"}),
                    json.dumps(
                        {
                            "predictions": [
                                {"id": "/B", "material": "__UNKNOWN__"},
                                {"/C": "Rubber"},
                            ]
                        }
                    ),
                    "{bad",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        mapping = task._load_prim_material_mapping(predictions_path)
        counts = task._count_prediction_materials(predictions_path)

        assert mapping == {
            "/A": "Steel",
            "/B": FALLBACK_MATERIAL_NAME,
            "/C": "Rubber",
        }
        assert counts["total"] == 3
        assert counts["actionable"] == 2
        assert counts["unknown"] == 1

    def test_defensive_no_resolved_material_branches(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        input_usd = tmp_path / "input.usda"
        stage = Usd.Stage.CreateNew(str(input_usd))
        UsdGeom.Xform.Define(stage, "/World")
        stage.GetRootLayer().Save()

        unknown_predictions = tmp_path / "unknown.jsonl"
        unknown_predictions.write_text(
            json.dumps({"id": "/World", "materials": {"material": "__UNKNOWN__"}})
            + "\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(
            apply_module,
            "material_mapping_with_fallback",
            lambda _mapping: {},
        )

        allowed = ApplyMaterialsToUSDTask().run(
            {
                "input_usd_path": str(input_usd),
                "output_usd_path": str(tmp_path / "allowed.usda"),
                "predictions_path": str(unknown_predictions),
                "resolved_materials": {},
                "allow_empty_predictions": True,
            }
        )
        assert allowed["assignment_stats"]["unknown"] == 1
        assert allowed["assignment_stats"]["unbound_prim_ids"] == ["/World"]

        with pytest.raises(ValueError, match="no usable visual evidence"):
            ApplyMaterialsToUSDTask().run(
                {
                    "input_usd_path": str(input_usd),
                    "output_usd_path": str(tmp_path / "blocked.usda"),
                    "predictions_path": str(unknown_predictions),
                    "resolved_materials": {},
                }
            )

        missing_predictions = tmp_path / "missing_material.jsonl"
        missing_predictions.write_text(
            json.dumps({"id": "/World", "materials": {}}) + "\n",
            encoding="utf-8",
        )
        missing_allowed = ApplyMaterialsToUSDTask().run(
            {
                "input_usd_path": str(input_usd),
                "output_usd_path": str(tmp_path / "missing_allowed.usda"),
                "predictions_path": str(missing_predictions),
                "resolved_materials": {},
                "allow_empty_predictions": True,
            }
        )
        assert missing_allowed["assignment_stats"]["missing"] == 1
        assert missing_allowed["assignment_stats"]["unbound_prim_ids"] == ["/World"]

        with pytest.raises(ValueError, match="did not contain actionable"):
            ApplyMaterialsToUSDTask().run(
                {
                    "input_usd_path": str(input_usd),
                    "output_usd_path": str(tmp_path / "missing_blocked.usda"),
                    "predictions_path": str(missing_predictions),
                    "resolved_materials": {},
                }
            )

    def test_run_warnings_for_existing_unknown_and_missing_predictions(
        self, tmp_path: Path
    ) -> None:
        input_usd = tmp_path / "input.usda"
        stage = Usd.Stage.CreateNew(str(input_usd))
        root = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(root.GetPrim())
        stage.GetRootLayer().Save()

        predictions_path = tmp_path / "predictions.jsonl"
        predictions_path.write_text(
            json.dumps({"id": "/World", "materials": {}}) + "\n",
            encoding="utf-8",
        )

        result = ApplyMaterialsToUSDTask().run(
            {
                "input_usd_path": str(input_usd),
                "output_usd_path": str(tmp_path / "output.usda"),
                "predictions_path": str(predictions_path),
                "resolved_materials": {"Steel": "steel.usd"},
                "unknown_material_predictions": "not-int",
                "allow_empty_predictions": True,
            }
        )

        assert result["unknown_material_predictions"] == 0
        assert result["assignment_stats"]["total_prims"] == 0

        with pytest.raises(ValueError, match="No material bindings"):
            ApplyMaterialsToUSDTask().run(
                {
                    "input_usd_path": str(input_usd),
                    "output_usd_path": str(tmp_path / "unknown_only.usda"),
                    "predictions_path": str(tmp_path / "empty.jsonl"),
                    "resolved_materials": {"Steel": "steel.usd"},
                    "unknown_material_predictions": 2,
                }
            )

    def test_apply_materials_to_instances_with_fake_stage_edges(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        class FakePath:
            def __init__(self, value: str):
                self.value = value

            def __str__(self) -> str:
                return self.value

        class FakePrototype:
            def __init__(self, path: str, valid: bool = True):
                self.path = path
                self.valid = valid

            def IsValid(self) -> bool:
                return self.valid

            def GetPath(self) -> FakePath:
                return FakePath(self.path)

        class FakePrim:
            def __init__(
                self,
                path: str,
                *,
                is_instance: bool,
                prototype: FakePrototype | None = None,
            ):
                self.path = path
                self.is_instance = is_instance
                self.prototype = prototype

            def GetPath(self) -> FakePath:
                return FakePath(self.path)

            def IsInstance(self) -> bool:
                return self.is_instance

            def GetPrototype(self) -> FakePrototype | None:
                return self.prototype

        class FakeStage:
            def __init__(self, prims: list[FakePrim]):
                self.prims = prims

            def Traverse(self) -> list[FakePrim]:
                return self.prims

        monkeypatch.setattr(apply_module, "nullify_material", lambda _prim: None)
        bound_paths: list[str] = []
        monkeypatch.setattr(
            task,
            "_author_material_binding_on_stage",
            lambda _stage, prim_path, material_path: bound_paths.append(
                f"{prim_path}->{material_path}"
            ),
        )
        stats = task._apply_materials_to_instances(
            FakeStage(
                [
                    FakePrim("/Already", is_instance=True),
                    FakePrim(
                        "/NoPrototype",
                        is_instance=True,
                        prototype=FakePrototype("/ProtoMissing", valid=False),
                    ),
                    FakePrim(
                        "/NoPrediction",
                        is_instance=True,
                        prototype=FakePrototype("/ProtoNoPrediction"),
                    ),
                    FakePrim(
                        "/NoMaterial",
                        is_instance=True,
                        prototype=FakePrototype("/ProtoNoMaterial"),
                    ),
                    FakePrim(
                        "/Apply",
                        is_instance=True,
                        prototype=FakePrototype("/ProtoApply"),
                    ),
                ]
            ),
            {
                "/Already": "Steel",
                "/ProtoNoMaterial": "MissingMaterial",
                "/ProtoApply": "Steel",
            },
            {"Steel": "/Materials/Steel"},
        )

        assert stats == {
            "instances_found": 4,
            "instances_applied": 1,
            "instances_skipped": 3,
        }
        assert bound_paths == ["/Apply->/Materials/Steel"]

        def fail_bind(_stage, _prim_path, _material_path):
            raise RuntimeError("bind failed")

        monkeypatch.setattr(task, "_author_material_binding_on_stage", fail_bind)
        failed_stats = task._apply_materials_to_instances(
            FakeStage(
                [
                    FakePrim(
                        "/Apply",
                        is_instance=True,
                        prototype=FakePrototype("/ProtoApply"),
                    )
                ]
            ),
            {"/ProtoApply": "Steel"},
            {"Steel": "/Materials/Steel"},
        )
        assert failed_stats["instances_skipped"] == 1

    def test_instance_remap_and_library_copy_error_edges(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()
        assert task._remap_instance_binding_target(
            "/Other/Mesh",
            {"/Instance": "/Proto"},
        ) == ("/Other/Mesh", False, False)
        assert task._remap_instance_binding_target(
            "/Instance/Mesh",
            {"/Instance": None},
        ) == ("/Instance/Mesh", False, True)

        assignments, remapped, skipped = task._group_binding_assignments(
            {
                "/InstanceA/Mesh": "Steel",
                "/InstanceB/Mesh": "Steel",
            },
            {
                "/InstanceA": "/Prototype",
                "/InstanceB": "/Prototype",
            },
        )
        assert assignments == [
            (
                "/Prototype/Mesh",
                "Steel",
                ["/InstanceA/Mesh", "/InstanceB/Mesh"],
                True,
            )
        ]
        assert remapped == 2
        assert skipped == 0

        conflicting, remapped, skipped = task._group_binding_assignments(
            {
                "/InstanceA/Mesh": "Steel",
                "/InstanceB/Mesh": "Wood",
            },
            {
                "/InstanceA": "/Prototype",
                "/InstanceB": "/Prototype",
            },
        )
        assert conflicting == []
        assert remapped == 2
        assert skipped == 0
        assert (
            "Leaving all source prims unbound"
            in task.listener.warning.call_args.args[0]
        )

        stage = Usd.Stage.CreateNew(str(tmp_path / "out.usda"))
        stage, copied = task._copy_library_materials(
            stage,
            str(tmp_path / "missing_library.usda"),
            tmp_path / "out.usda",
            {"Steel": "/World/Looks/Steel"},
        )
        assert copied == {}

        library_path = tmp_path / "library.usda"
        library_stage = Usd.Stage.CreateNew(str(library_path))
        UsdShade.Material.Define(library_stage, "/World/Looks/Steel")
        library_stage.GetRootLayer().Save()

        with monkeypatch.context() as scoped:
            scoped.setattr(Sdf.Layer, "FindOrOpen", staticmethod(lambda _path: None))
            _stage, copied = task._copy_library_materials(
                stage,
                str(library_path),
                tmp_path / "out.usda",
                {"Steel": "/World/Looks/Steel"},
            )
            assert copied == {}

        _stage, copied = task._copy_library_materials(
            stage,
            str(library_path),
            tmp_path / "out.usda",
            {FALLBACK_MATERIAL_NAME: "/World/Looks/MissingFallback"},
        )
        assert copied[FALLBACK_MATERIAL_NAME] == "/World/Looks/MissingFallback"

        _stage, copied = task._copy_library_materials(
            stage,
            str(library_path),
            tmp_path / "out.usda",
            {"Missing": "/World/Looks/Missing"},
        )
        assert copied == {}

        with monkeypatch.context() as scoped:
            scoped.setattr(Sdf, "CopySpec", lambda *_args, **_kwargs: False)
            _stage, copied = task._copy_library_materials(
                stage,
                str(library_path),
                tmp_path / "out.usda",
                {"Steel": "/World/Looks/Steel"},
            )
            assert copied == {}

    def test_low_level_asset_and_graph_helper_edges(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        class FakeTargetPathList:
            explicitItems = []

            def ClearEditsAndMakeExplicit(self):
                return None

        class FakeRelationship:
            custom = True
            targetPathList = FakeTargetPathList()

        class FakePrimSpec:
            relationships = {"material:binding": FakeRelationship()}

            def __init__(self):
                self.info = {}

            def GetInfo(self, name):
                return self.info.get(name)

            def SetInfo(self, name, value):
                self.info[name] = value

        fake_prim_spec = FakePrimSpec()
        fake_root_layer = MagicMock()
        fake_root_layer.GetPrimAtPath.return_value = fake_prim_spec
        task._author_material_binding_in_layer(
            fake_root_layer,
            "/World/Mesh",
            "/World/Looks/Steel",
        )
        assert "apiSchemas" in fake_prim_spec.info

        binding_stage = Usd.Stage.CreateNew(str(tmp_path / "binding.usda"))
        task._author_material_binding_in_layer(
            binding_stage.GetRootLayer(),
            "/World/Mesh",
            "/World/Looks/Steel",
        )
        assert binding_stage.GetRootLayer().GetPrimAtPath("/World/Mesh")

        class BadAssetValue:
            @property
            def path(self):
                raise RuntimeError("bad path")

            def __str__(self) -> str:
                return "fallback.mdl"

        assert task._asset_path_to_string(BadAssetValue()) == "fallback.mdl"
        assert not task._is_unresolved_local_asset_path(object(), "", tmp_path)
        with monkeypatch.context() as scoped:
            scoped.setattr(
                Path, "exists", MagicMock(side_effect=OSError("stat failed"))
            )
            assert task._is_unresolved_local_asset_path(
                type("Asset", (), {"resolvedPath": ""})(),
                "missing.mdl",
                tmp_path,
            )

        stage = Usd.Stage.CreateNew(str(tmp_path / "helpers.usda"))
        root = UsdGeom.Xform.Define(stage, "/World")
        material = UsdShade.Material.Define(stage, "/World/Looks/Mat")
        shader = UsdShade.Shader.Define(stage, "/World/Looks/Mat/Shader")
        null_shader = UsdShade.Shader.Define(stage, "/World/Looks/Mat/NullShader")
        null_shader.GetPrim().CreateAttribute(
            "info:mdl:sourceAsset",
            Sdf.ValueTypeNames.Asset,
        )
        xform = UsdGeom.Xform.Define(stage, "/World/Xform")
        ref_prim = UsdGeom.Xform.Define(stage, "/World/RefPrim")
        ref_prim.GetPrim().GetReferences().AddInternalReference("/World/Looks/Mat")
        ref_prim.GetPrim().GetPayloads().AddInternalPayload("/World/Looks/Mat")
        ref_prim.GetPrim().GetPayloads().AddPayload(
            "external.usda",
            "/World/Looks/ExternalMat",
        )

        assert (
            task._collect_unresolved_mdl_shader_paths_from_prims(
                stage,
                {
                    "/Missing",
                    str(root.GetPath()),
                    str(xform.GetPath()),
                    str(shader.GetPath()),
                    str(null_shader.GetPath()),
                },
            )
            == []
        )
        assert (
            task._collect_unresolved_mdl_shader_paths(stage, material.GetPrim()) == []
        )
        assert {
            str(path) for path in task._composition_target_paths(ref_prim.GetPrim())
        } == {"/World/Looks/Mat"}
        assert (
            task._collect_composition_target_material_paths(
                stage,
                [str(material.GetPath()), str(material.GetPath())],
                protected_material_paths={str(material.GetPath())},
            )
            == set()
        )
        assert (
            task._collect_composition_target_material_paths(
                stage,
                {"/Missing"},
            )
            == set()
        )
        monkeypatch.setattr(
            task,
            "_collect_material_graph_prim_paths",
            lambda _stage, _prim: {"/MissingPrim"},
        )
        assert (
            task._collect_composition_target_material_paths(
                stage,
                {str(material.GetPath())},
            )
            == set()
        )
        assert task._collect_connected_material_paths(stage, {"/Missing"}) == set()
        assert (
            task._collect_reachable_shader_prim_paths(stage, {"/Missing"}, set())
            == set()
        )

    def test_graph_traversal_defensive_edges(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        task = ApplyMaterialsToUSDTask()

        class FakeConnection:
            def __init__(self, prim_path):
                self._prim_path = prim_path

            def GetPrimPath(self):
                return self._prim_path

        class FakeAttr:
            def __init__(self, connections):
                self._connections = connections

            def GetConnections(self):
                return self._connections

        class FakePrim:
            def __init__(
                self,
                paths,
                *,
                valid_values=(True,),
                active_values=(True,),
                attrs=(),
            ):
                self._paths = list(paths) if isinstance(paths, list) else [paths]
                self._valid_values = list(valid_values)
                self._active_values = list(active_values)
                self._attrs = list(attrs)

            def _next(self, values):
                if len(values) > 1:
                    return values.pop(0)
                return values[0]

            def IsValid(self):
                return self._next(self._valid_values)

            def IsActive(self):
                return self._next(self._active_values)

            def GetPath(self):
                if len(self._paths) > 1:
                    return self._paths.pop(0)
                return self._paths[0]

            def GetAttributes(self):
                return self._attrs

        graph_material = FakePrim(
            "/Mat",
            attrs=[FakeAttr([FakeConnection(Sdf.Path.emptyPath)])],
        )
        graph_stage = MagicMock()
        graph_stage.GetPrimAtPath.side_effect = lambda path: {
            Sdf.Path("/InvalidQueue"): FakePrim("/InvalidQueue", valid_values=(False,)),
            Sdf.Path("/LoopInactive"): FakePrim(
                "/LoopInactive",
                valid_values=(True, False),
            ),
            Sdf.Path("/QueuedDuplicate"): FakePrim(["/QueuedDuplicate", "/Mat"]),
        }[path]
        monkeypatch.setattr(
            task,
            "_composition_target_paths",
            lambda _prim: [
                Sdf.Path("/InvalidQueue"),
                Sdf.Path("/LoopInactive"),
                Sdf.Path("/QueuedDuplicate"),
            ],
        )

        assert task._collect_material_graph_prim_paths(graph_stage, graph_material) == {
            "/Mat"
        }

        connected_stage = MagicMock()
        connected_prims = {
            "/Root": FakePrim("/Root"),
            "/InvalidQueue": FakePrim("/InvalidQueue", valid_values=(False,)),
            "/LoopInactive": FakePrim(
                "/LoopInactive",
                valid_values=(True, False),
            ),
            "/QueuedDuplicate": FakePrim(["/QueuedDuplicate", "/First"]),
            "/First": FakePrim(
                "/First",
                attrs=[FakeAttr([FakeConnection(Sdf.Path.emptyPath)])],
            ),
        }
        connected_stage.GetPrimAtPath.side_effect = connected_prims.__getitem__
        monkeypatch.setattr(
            task,
            "_collect_material_graph_prim_paths",
            lambda *_args: [
                "/LoopInactive",
                "/QueuedDuplicate",
                "/First",
                "/InvalidQueue",
            ],
        )

        assert (
            task._collect_connected_material_paths(
                connected_stage,
                {"/Root"},
                protected_material_paths={"/Root"},
            )
            == set()
        )

    def test_deactivate_unresolved_mdl_iteration_and_duplicate_edges(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()
        stage = Usd.Stage.CreateNew(str(tmp_path / "cleanup.usda"))
        material = UsdShade.Material.Define(stage, "/World/Looks/Mat")
        shader = UsdShade.Shader.Define(stage, "/World/Looks/Mat/Shader")

        monkeypatch.setattr(
            task, "_collect_bound_material_paths", lambda _stage: {"/Seed"}
        )
        counter = {"value": 0}

        def keep_discovering(*_args, **_kwargs):
            counter["value"] += 1
            return {f"/Discovered{counter['value']}"}

        monkeypatch.setattr(
            task,
            "_collect_composition_target_material_paths",
            keep_discovering,
        )
        monkeypatch.setattr(
            task, "_collect_connected_material_paths", lambda *_a, **_k: set()
        )
        monkeypatch.setattr(
            task, "_collect_reachable_shader_prim_paths", lambda *_a, **_k: set()
        )
        monkeypatch.setattr(
            task, "_collect_unresolved_mdl_shader_paths", lambda *_a, **_k: []
        )
        monkeypatch.setattr(
            task,
            "_collect_unresolved_mdl_shader_paths_from_prims",
            lambda *_a, **_k: [],
        )

        assert task._deactivate_unbound_unresolved_mdl_shaders(stage) == []
        task.listener.warning.assert_any_call(
            "Stopping stale MDL material reachability traversal after "
            "3 iterations; remaining frontier: /Discovered3"
        )

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()
        monkeypatch.setattr(task, "_collect_bound_material_paths", lambda _stage: set())
        monkeypatch.setattr(
            task, "_collect_reachable_shader_prim_paths", lambda *_a, **_k: set()
        )
        monkeypatch.setattr(
            task, "_collect_material_graph_prim_paths", lambda *_a, **_k: set()
        )
        monkeypatch.setattr(
            task,
            "_collect_unresolved_mdl_shader_paths",
            lambda *_a, **_k: [str(shader.GetPath()), str(shader.GetPath())],
        )
        monkeypatch.setattr(
            task,
            "_collect_unresolved_mdl_shader_paths_from_prims",
            lambda *_a, **_k: [],
        )

        assert task._deactivate_unbound_unresolved_mdl_shaders(stage) == [
            str(shader.GetPath())
        ]
        shader.GetPrim().SetActive(True)

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()
        monkeypatch.setattr(task, "_collect_bound_material_paths", lambda _stage: set())
        monkeypatch.setattr(
            task,
            "_collect_composition_target_material_paths",
            lambda *_a, **_k: set(),
        )
        monkeypatch.setattr(
            task, "_collect_connected_material_paths", lambda *_a, **_k: set()
        )
        monkeypatch.setattr(
            task, "_collect_reachable_shader_prim_paths", lambda *_a, **_k: set()
        )
        monkeypatch.setattr(
            task, "_collect_material_graph_prim_paths", lambda *_a, **_k: set()
        )
        monkeypatch.setattr(
            task,
            "_collect_unresolved_mdl_shader_paths",
            lambda *_a, **_k: [str(shader.GetPath()), str(shader.GetPath())],
        )
        monkeypatch.setattr(
            task,
            "_collect_unresolved_mdl_shader_paths_from_prims",
            lambda *_a, **_k: [],
        )

        assert task._deactivate_unbound_unresolved_mdl_shaders(
            stage,
            protected_material_paths={str(material.GetPath())},
        ) == [str(shader.GetPath())]

    def test_full_stage_instance_proxy_and_flatten_open_edges(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        input_usd = tmp_path / "input_instance.usda"
        input_stage = Usd.Stage.CreateNew(str(input_usd))
        world = UsdGeom.Xform.Define(input_stage, "/World")
        input_stage.SetDefaultPrim(world.GetPrim())
        UsdGeom.Mesh.Define(input_stage, "/Proto/Mesh")
        instance = UsdGeom.Xform.Define(input_stage, "/World/Inst")
        instance.GetPrim().GetReferences().AddInternalReference("/Proto")
        instance.GetPrim().SetInstanceable(True)
        input_stage.GetRootLayer().Save()

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()
        monkeypatch.setattr(
            task,
            "_create_material_on_stage",
            lambda **_kwargs: ("/Materials/Steel", True, {}),
        )
        monkeypatch.setattr(
            task, "_get_local_instance_reference_map", lambda _stage: {}
        )

        _stage, _materials_applied, stats = task._create_full_stage(
            input_usd,
            tmp_path / "output_instance.usda",
            {"Steel": "steel.usd"},
            {"/World/Inst/Mesh": "Steel"},
            skip_instance_check=True,
        )

        assert stats["prims_with_materials"] == 0
        task.listener.info.assert_any_call("Skipped 1 instance proxy prims (read-only)")

        original_open = Usd.Stage.Open

        def open_or_none(value):
            if isinstance(value, Sdf.Layer):
                return None
            return original_open(value)

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()
        monkeypatch.setattr(Usd.Stage, "Open", staticmethod(open_or_none))
        monkeypatch.setattr(
            task,
            "_create_material_on_stage",
            lambda **_kwargs: ("/Materials/Steel", True, {}),
        )
        monkeypatch.setattr(apply_module, "nullify_material", lambda _prim: None)
        monkeypatch.setattr(
            task,
            "_author_material_binding_on_stage",
            lambda *_args, **_kwargs: None,
        )

        _stage, _materials_applied, stats = task._create_full_stage(
            input_usd,
            tmp_path / "output_flatten_none.usda",
            {"Steel": "steel.usd"},
            {"/World": "Steel"},
            flatten_output=True,
            skip_instance_check=True,
        )

        assert stats["prims_with_materials"] == 1
        task.listener.warning.assert_any_call(
            "Failed to open flattened layer for stale MDL cleanup; "
            "continuing with export"
        )

    def test_full_stage_binding_edge_paths(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        input_usd = tmp_path / "input_full.usda"
        input_stage = Usd.Stage.CreateNew(str(input_usd))
        root = UsdGeom.Xform.Define(input_stage, "/World")
        input_stage.SetDefaultPrim(root.GetPrim())
        mesh = UsdGeom.Mesh.Define(input_stage, "/World/Mesh")
        subset = UsdGeom.Subset.Define(input_stage, "/World/Mesh/Subset")
        subset.CreateElementTypeAttr(UsdGeom.Tokens.face)
        UsdGeom.Mesh.Define(input_stage, "/World/Falsy")
        input_stage.GetRootLayer().Save()

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()
        monkeypatch.setattr(
            task,
            "_create_material_on_stage",
            lambda **kwargs: (
                "" if kwargs["material_name"] == "Falsy" else "/Materials/Steel",
                True,
                {},
            ),
        )
        monkeypatch.setattr(
            task,
            "_get_local_instance_reference_map",
            lambda _stage: {"/External": None},
        )
        monkeypatch.setattr(apply_module, "nullify_material", lambda _prim: None)
        bound: list[str] = []
        monkeypatch.setattr(
            task,
            "_author_material_binding_on_stage",
            lambda _stage, prim_path, material_path: bound.append(
                f"{prim_path}->{material_path}"
            ),
        )

        _stage, materials_applied, stats = task._create_full_stage(
            input_usd,
            tmp_path / "output_full.usda",
            {"Steel": "steel.usd", "Falsy": "falsy.usd"},
            {
                "/External/Mesh": "Steel",
                "/World/Mesh": "Steel",
                "/World/Falsy": "Falsy",
            },
            skip_instance_check=True,
        )

        assert materials_applied["Falsy"] == ""
        assert stats["prims_with_materials"] == 1
        assert bound == ["/World/Mesh->/Materials/Steel"]
        assert mesh

    def test_material_layer_library_file_display_and_skip_edges(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        input_usd = tmp_path / "input_layer.usda"
        input_stage = Usd.Stage.CreateNew(str(input_usd))
        root = UsdGeom.Xform.Define(input_stage, "/World")
        input_stage.SetDefaultPrim(root.GetPrim())
        UsdGeom.Mesh.Define(input_stage, "/World/Mesh")
        input_stage.GetRootLayer().Save()

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()
        monkeypatch.setattr(
            task,
            "_copy_library_materials",
            lambda stage, *_args, **_kwargs: (
                stage,
                {"Steel": "/World/Looks/Steel"},
            ),
        )
        monkeypatch.setattr(
            task,
            "_add_preview_surface_overlays_for_mdl_materials",
            lambda *_args, **_kwargs: 1,
        )
        monkeypatch.setattr(
            apply_module,
            "add_ovrtx_preview_fallbacks_for_materialx_openpbr",
            lambda *_args, **_kwargs: 1,
        )
        _stage, materials_applied, stats = task._create_material_layer(
            input_usd,
            tmp_path / "library_layer.usda",
            {"Steel": "/Library/Steel"},
            {"/World/Mesh": "Steel"},
            is_library_based=True,
            material_library_path="library.usda",
            material_profile="preview_surface",
        )
        assert materials_applied == {"Steel": "/World/Looks/Steel"}
        assert stats["prims_with_materials"] == 1

        monkeypatch.setattr(
            task,
            "_create_material_on_stage",
            lambda **kwargs: (f"/Materials/{kwargs['material_name']}", True, {}),
        )
        monkeypatch.setattr(
            task,
            "_get_local_instance_reference_map",
            lambda _stage: {"/External": None},
        )
        _stage, materials_applied, stats = task._create_material_layer(
            input_usd,
            tmp_path / "file_layer.usda",
            {"Steel": "steel.usd"},
            {
                "/External/Mesh": "Steel",
                "/World/Mesh": "Steel",
                "/World/Missing": "Missing",
            },
            skip_instance_check=True,
        )
        assert materials_applied == {"Steel": "/Materials/Steel"}
        assert stats["prims_with_materials"] == 1

        _stage, materials_applied, stats = task._create_material_layer(
            input_usd,
            tmp_path / "display_layer.usda",
            {"Steel": "ignored"},
            {"/World/Mesh": "Steel"},
            material_profile="display_color",
        )
        assert materials_applied == {"Steel": "display_color"}
        assert stats["prims_with_materials"] == 1

        monkeypatch.setattr(
            task,
            "_create_material_on_stage",
            lambda **kwargs: (
                "" if kwargs["material_name"] == "Empty" else "/Materials/Steel",
                True,
                {},
            ),
        )
        monkeypatch.setattr(
            task,
            "_get_local_instance_reference_map",
            lambda _stage: {"/Instance": "/Prototype"},
        )
        _stage, materials_applied, stats = task._create_material_layer(
            input_usd,
            tmp_path / "remapped_layer.usda",
            {"Steel": "steel.usd", "Empty": "empty.usd"},
            {
                "/Instance/Good": "Steel",
                "/Instance/MissingMaterialPath": "Empty",
            },
            skip_instance_check=True,
        )
        assert materials_applied == {"Steel": "/Materials/Steel", "Empty": ""}
        assert stats["prims_with_materials"] == 1
