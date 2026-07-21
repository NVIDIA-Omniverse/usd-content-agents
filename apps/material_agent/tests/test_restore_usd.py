# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for RestoreUSDTask with all 8 scene optimizer operation combinations.

Tests cover:
- no_op: No operations, identity mapping
- D_only: Deinstancing only
- S_only: Split GeomSubsets only
- P_only: Deduplication only
- DS: Deinstance + Split
- DP: Deinstance + Dedup
- SP: Split + Dedup
- DSP: Deinstance + Split + Dedup (all operations)
"""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from material_agent.tasks.restore_usd import RestorationStats, RestoreUSDTask

# Path to regression test data
TEST_DATA_DIR = Path(__file__).parent.parent / "data" / "regression" / "scene_optimizer"
TEST_USDA = TEST_DATA_DIR / "scene_optimizer_test.usda"

# All 8 operation combinations
ALL_OPERATIONS = ["no_op", "D_only", "S_only", "P_only", "DS", "DP", "SP", "DSP"]


def load_correspondence_map(name: str) -> dict:
    """Load a correspondence map from test data."""
    path = TEST_DATA_DIR / f"correspondence_map_{name}.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # Wrap in optimization_metadata structure (as seen from optimize_usd output)
    return {"correspondence_map": data}


def get_unique_prototypes(optimization_metadata: dict) -> list[str]:
    """Extract unique prototype paths from a correspondence map."""
    o2p = (
        optimization_metadata.get("correspondence_map", {})
        .get("full_mapping", {})
        .get("original_to_prototype", {})
    )
    seen: set[str] = set()
    result: list[str] = []
    for proto_list in o2p.values():
        if not isinstance(proto_list, list):
            proto_list = [proto_list]
        for p in proto_list:
            if p not in seen:
                seen.add(p)
                result.append(p)
    return result


def make_predictions(prototype_paths: list[str]) -> list[dict]:
    """Create synthetic predictions for a list of prototype paths."""
    return [
        {
            "id": path,
            "material": f"mat_{path.rsplit('/', 1)[-1]}",
            "confidence": 0.9,
        }
        for path in prototype_paths
    ]


def write_predictions(predictions: list[dict], path: Path) -> None:
    """Write predictions to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")


def read_predictions(path: Path) -> list[dict]:
    """Read predictions from a JSONL file."""
    results = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def mock_listener() -> Mock:
    """Create a mock event listener."""
    listener = Mock()
    listener.info = Mock()
    listener.warning = Mock()
    listener.error = Mock()
    listener.debug = Mock()
    return listener


def run_restore(
    optimization_metadata: dict,
    predictions: list[dict],
    tmp_path: Path,
    original_usd_path: Path | None = None,
) -> tuple[list[dict], RestorationStats]:
    """Run RestoreUSDTask._transform_predictions and return results.

    Args:
        optimization_metadata: Optimization metadata with correspondence_map
        predictions: Input predictions to transform
        tmp_path: Temporary directory for I/O
        original_usd_path: Path to original USD (for GeomSubset inspection)

    Returns:
        Tuple of (output_predictions, stats)
    """
    task = RestoreUSDTask()
    listener = mock_listener()

    predictions_path = tmp_path / "input" / "predictions.jsonl"
    output_path = tmp_path / "output" / "restored_predictions.jsonl"
    write_predictions(predictions, predictions_path)

    if original_usd_path is None:
        original_usd_path = TEST_USDA

    count, stats = task._transform_predictions(
        predictions_path=predictions_path,
        output_predictions_path=output_path,
        original_usd_path=original_usd_path,
        optimization_metadata=optimization_metadata,
        listener=listener,
    )

    output_predictions = read_predictions(output_path) if output_path.exists() else []
    assert count == len(output_predictions)
    assert count == stats.predictions_written
    return output_predictions, stats


# ---------------------------------------------------------------------------
# Parametrized test: all 8 operation combinations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op_name", ALL_OPERATIONS)
def test_all_operations_produce_output(op_name: str, tmp_path: Path) -> None:
    """Every operation combination should produce at least one output prediction."""
    metadata = load_correspondence_map(op_name)
    prototypes = get_unique_prototypes(metadata)
    predictions = make_predictions(prototypes)

    output, stats = run_restore(metadata, predictions, tmp_path)

    assert stats.predictions_written > 0, f"No predictions written for {op_name}"
    assert stats.total_originals == 12, f"Expected 12 originals for {op_name}"
    assert len(stats.uncovered_originals) == 0, (
        f"Uncovered originals for {op_name}: {stats.uncovered_originals}"
    )


@pytest.mark.parametrize("op_name", ALL_OPERATIONS)
def test_all_predictions_consumed(op_name: str, tmp_path: Path) -> None:
    """Every input prediction should be consumed (mapped to at least one original)."""
    metadata = load_correspondence_map(op_name)
    prototypes = get_unique_prototypes(metadata)
    predictions = make_predictions(prototypes)

    _, stats = run_restore(metadata, predictions, tmp_path)

    assert len(stats.unconsumed_predictions) == 0, (
        f"Unconsumed predictions for {op_name}: {stats.unconsumed_predictions}"
    )


# ---------------------------------------------------------------------------
# Identity cases: no_op and D_only
# ---------------------------------------------------------------------------


class TestIdentityCases:
    """Test no_op and D_only where all mappings are 1:1 identity."""

    @pytest.mark.parametrize("op_name", ["no_op", "D_only"])
    def test_output_ids_match_originals(self, op_name: str, tmp_path: Path) -> None:
        """Output prediction IDs should equal original prim paths."""
        metadata = load_correspondence_map(op_name)
        prototypes = get_unique_prototypes(metadata)
        predictions = make_predictions(prototypes)

        output, stats = run_restore(metadata, predictions, tmp_path)

        o2p = metadata["correspondence_map"]["full_mapping"]["original_to_prototype"]
        output_ids = {p["id"] for p in output}

        # Every original should appear in output
        assert output_ids == set(o2p.keys())
        assert stats.identity_count == 12
        assert stats.dedup_count == 0
        assert stats.split_count == 0
        assert stats.split_dedup_count == 0


# ---------------------------------------------------------------------------
# Dedup cases: P_only and DP
# ---------------------------------------------------------------------------


class TestDedupCases:
    """Test P_only and DP where multiple originals share a prototype."""

    @pytest.mark.parametrize("op_name", ["P_only", "DP"])
    def test_dedup_expands_to_all_originals(self, op_name: str, tmp_path: Path) -> None:
        """Deduplicated pyramids should each get their own output prediction."""
        metadata = load_correspondence_map(op_name)
        prototypes = get_unique_prototypes(metadata)
        predictions = make_predictions(prototypes)

        output, stats = run_restore(metadata, predictions, tmp_path)

        output_ids = {p["id"] for p in output}

        # All 3 pyramids should appear with their original paths
        assert "/World/DeduplicationTest/DuplicatePyramid1" in output_ids
        assert "/World/DeduplicationTest/DuplicatePyramid2" in output_ids
        assert "/World/DeduplicationTest/DuplicatePyramid3" in output_ids

        # All 3 should share the same material (from the one prototype)
        pyramid_preds = [p for p in output if "DeduplicationTest" in p["id"]]
        materials = {p["material"] for p in pyramid_preds}
        assert len(materials) == 1, (
            "All deduplicated pyramids should share one material"
        )

        assert stats.dedup_count >= 2  # Pyramid2 and Pyramid3 are dedup mappings

    @pytest.mark.parametrize("op_name", ["P_only", "DP"])
    def test_non_dedup_prims_unchanged(self, op_name: str, tmp_path: Path) -> None:
        """Prims not affected by dedup should pass through unchanged."""
        metadata = load_correspondence_map(op_name)
        prototypes = get_unique_prototypes(metadata)
        predictions = make_predictions(prototypes)

        output, _ = run_restore(metadata, predictions, tmp_path)

        output_ids = {p["id"] for p in output}
        # Control group should be identity
        assert "/World/ControlGroup/SimpleMesh" in output_ids


# ---------------------------------------------------------------------------
# Split cases: S_only and DS
# ---------------------------------------------------------------------------


class TestSplitCases:
    """Test S_only and DS where one original maps to multiple split parts."""

    @pytest.mark.parametrize("op_name", ["S_only", "DS"])
    def test_split_produces_geomsubset_ids(self, op_name: str, tmp_path: Path) -> None:
        """Split prims should produce predictions with GeomSubset child paths."""
        metadata = load_correspondence_map(op_name)
        prototypes = get_unique_prototypes(metadata)
        predictions = make_predictions(prototypes)

        output, stats = run_restore(metadata, predictions, tmp_path)

        # TwoPrismsMesh should be split into GeomSubset children
        prism_preds = [p for p in output if "SplitGeomSubsetsTest" in p["id"]]
        assert len(prism_preds) == 2, "TwoPrismsMesh should produce 2 predictions"

        # Should have GeomSubset paths (from the original USD)
        prism_ids = {p["id"] for p in prism_preds}
        assert "/World/SplitGeomSubsetsTest/TwoPrismsMesh/Prism_Left" in prism_ids
        assert "/World/SplitGeomSubsetsTest/TwoPrismsMesh/Prism_Right" in prism_ids

        assert stats.split_count >= 1

    @pytest.mark.parametrize("op_name", ["S_only", "DS"])
    def test_split_arrow_meshes(self, op_name: str, tmp_path: Path) -> None:
        """Arrow meshes should each be split into head and body GeomSubsets."""
        metadata = load_correspondence_map(op_name)
        prototypes = get_unique_prototypes(metadata)
        predictions = make_predictions(prototypes)

        output, _ = run_restore(metadata, predictions, tmp_path)

        # Each arrow mesh should produce 2 predictions (head + body)
        for mesh_name in ["ArrowMesh1", "ArrowMesh2", "ArrowMesh3"]:
            arrow_preds = [p for p in output if mesh_name in p["id"]]
            assert len(arrow_preds) == 2, (
                f"{mesh_name} should produce 2 predictions, got {len(arrow_preds)}"
            )

    @pytest.mark.parametrize("op_name", ["S_only", "DS"])
    def test_combined_test_split(self, op_name: str, tmp_path: Path) -> None:
        """CombinedTest PrototypeDoubleTetra should split into 2 GeomSubsets."""
        metadata = load_correspondence_map(op_name)
        prototypes = get_unique_prototypes(metadata)
        predictions = make_predictions(prototypes)

        output, _ = run_restore(metadata, predictions, tmp_path)

        combined_preds = [p for p in output if "CombinedTest" in p["id"]]
        assert len(combined_preds) == 2
        combined_ids = {p["id"] for p in combined_preds}
        assert "/World/CombinedTest/PrototypeDoubleTetra/Tetra_Left" in combined_ids
        assert "/World/CombinedTest/PrototypeDoubleTetra/Tetra_Right" in combined_ids


# ---------------------------------------------------------------------------
# Split + Dedup cases: SP and DSP
# ---------------------------------------------------------------------------


class TestSplitDedupCases:
    """Test SP and DSP where split parts are also deduplicated."""

    @pytest.mark.parametrize("op_name", ["SP", "DSP"])
    def test_split_dedup_arrows_all_get_predictions(
        self, op_name: str, tmp_path: Path
    ) -> None:
        """All 3 arrow meshes should get predictions even though prototypes are shared."""
        metadata = load_correspondence_map(op_name)
        prototypes = get_unique_prototypes(metadata)
        predictions = make_predictions(prototypes)

        output, stats = run_restore(metadata, predictions, tmp_path)

        # Each arrow mesh should produce 2 predictions
        for mesh_name in ["ArrowMesh1", "ArrowMesh2", "ArrowMesh3"]:
            arrow_preds = [p for p in output if mesh_name in p["id"]]
            assert len(arrow_preds) == 2, (
                f"{mesh_name} should produce 2 predictions for {op_name}"
            )

        # ArrowMesh2 and ArrowMesh3 share prototypes with ArrowMesh1
        # so all should have the same materials
        arrow1_materials = sorted(
            p["material"] for p in output if "ArrowMesh1" in p["id"]
        )
        arrow2_materials = sorted(
            p["material"] for p in output if "ArrowMesh2" in p["id"]
        )
        arrow3_materials = sorted(
            p["material"] for p in output if "ArrowMesh3" in p["id"]
        )
        assert arrow1_materials == arrow2_materials
        assert arrow1_materials == arrow3_materials

    @pytest.mark.parametrize("op_name", ["SP", "DSP"])
    def test_duplicate_prototype_entries_handled(
        self, op_name: str, tmp_path: Path
    ) -> None:
        """Handle cases where prototype list has duplicate entries (split parts deduped to same proto)."""
        metadata = load_correspondence_map(op_name)
        o2p = metadata["correspondence_map"]["full_mapping"]["original_to_prototype"]

        # Verify the test data has the expected duplicate entries
        double_tetra_protos = o2p.get("/World/CombinedTest/PrototypeDoubleTetra", [])
        assert len(double_tetra_protos) == 2
        # Both entries point to the same path (deduped)
        assert double_tetra_protos[0] == double_tetra_protos[1]

        prototypes = get_unique_prototypes(metadata)
        predictions = make_predictions(prototypes)

        output, stats = run_restore(metadata, predictions, tmp_path)

        # Should still produce 2 predictions for the 2 GeomSubsets
        combined_preds = [p for p in output if "CombinedTest" in p["id"]]
        assert len(combined_preds) == 2

        assert stats.split_dedup_count >= 1

    @pytest.mark.parametrize("op_name", ["SP", "DSP"])
    def test_dedup_pyramids_in_split_dedup_mode(
        self, op_name: str, tmp_path: Path
    ) -> None:
        """Pyramids should still be dedup-expanded even in SP/DSP mode."""
        metadata = load_correspondence_map(op_name)
        prototypes = get_unique_prototypes(metadata)
        predictions = make_predictions(prototypes)

        output, stats = run_restore(metadata, predictions, tmp_path)

        output_ids = {p["id"] for p in output}
        assert "/World/DeduplicationTest/DuplicatePyramid1" in output_ids
        assert "/World/DeduplicationTest/DuplicatePyramid2" in output_ids
        assert "/World/DeduplicationTest/DuplicatePyramid3" in output_ids


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_metadata_passthrough(self, tmp_path: Path) -> None:
        """Empty optimization metadata should copy predictions as-is."""
        predictions = [
            {"id": "/some/path", "material": "steel"},
            {"id": "/other/path", "material": "wood"},
        ]
        output, stats = run_restore({}, predictions, tmp_path)

        assert len(output) == 2
        assert output[0]["id"] == "/some/path"
        assert output[1]["id"] == "/other/path"

    def test_empty_predictions_file(self, tmp_path: Path) -> None:
        """Empty predictions file should produce empty output without errors."""
        metadata = load_correspondence_map("no_op")
        output, stats = run_restore(metadata, [], tmp_path)

        assert len(output) == 0
        assert stats.predictions_written == 0
        assert len(stats.uncovered_originals) == 12  # All originals uncovered

    def test_missing_prediction_for_prototype(self, tmp_path: Path) -> None:
        """Missing predictions should log warnings and still process others."""
        metadata = load_correspondence_map("no_op")
        # Only provide predictions for a subset
        predictions = [
            {"id": "/World/ControlGroup/SimpleMesh", "material": "steel"},
        ]
        output, stats = run_restore(metadata, predictions, tmp_path)

        assert len(output) == 1
        assert output[0]["id"] == "/World/ControlGroup/SimpleMesh"
        assert len(stats.uncovered_originals) == 11  # 11 originals without predictions

    def test_unconsumed_predictions_tracked(self, tmp_path: Path) -> None:
        """Predictions not matching any mapping should be tracked."""
        metadata = load_correspondence_map("no_op")
        prototypes = get_unique_prototypes(metadata)
        predictions = make_predictions(prototypes)
        # Add an extra prediction that doesn't match any original
        predictions.append({"id": "/Nonexistent/Path", "material": "ghost"})

        _, stats = run_restore(metadata, predictions, tmp_path)

        assert "/Nonexistent/Path" in stats.unconsumed_predictions

    def test_no_stage_fallback_for_split(self, tmp_path: Path) -> None:
        """When USD stage is unavailable, split should use indexed fallback paths."""
        metadata = load_correspondence_map("S_only")
        prototypes = get_unique_prototypes(metadata)
        predictions = make_predictions(prototypes)

        # Use a non-existent USD path so stage opening fails
        fake_usd = tmp_path / "nonexistent.usda"
        output, stats = run_restore(
            metadata, predictions, tmp_path, original_usd_path=fake_usd
        )

        # Split prisms should use indexed fallback paths
        prism_preds = [p for p in output if "SplitGeomSubsetsTest" in p["id"]]
        assert len(prism_preds) == 2
        prism_ids = sorted(p["id"] for p in prism_preds)
        assert prism_ids[0].endswith("_part_0")
        assert prism_ids[1].endswith("_part_1")


class TestRestorationStats:
    """Test the RestorationStats dataclass."""

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        stats = RestorationStats(
            total_originals=10,
            identity_count=5,
            dedup_count=3,
            split_count=1,
            split_dedup_count=1,
            predictions_consumed={"a", "b", "c"},
            predictions_written=10,
            uncovered_originals=["/missing"],
            unconsumed_predictions=["/extra"],
        )
        d = stats.to_dict()

        assert d["total_originals"] == 10
        assert d["identity_count"] == 5
        assert d["dedup_count"] == 3
        assert d["split_count"] == 1
        assert d["split_dedup_count"] == 1
        assert d["predictions_consumed"] == 3  # len of set
        assert d["predictions_written"] == 10
        assert d["uncovered_originals"] == ["/missing"]
        assert d["unconsumed_predictions"] == ["/extra"]
        assert d["restored_prim_sources"] == {}
        assert d["expected_target_count"] == 0
        assert d["mapping_complete"] is True
        assert d["mapping_warnings"] == []
