# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Material Agent API types."""

import pytest

from material_agent.api.types import (
    APIResult,
    AssignmentStats,
    DownloadStats,
    MaterialSearchResult,
    MetricsResult,
)


class TestAPIResult:
    """Tests for APIResult base class."""

    def test_api_result_success(self):
        """Test creating a successful APIResult."""
        result = APIResult(success=True)
        assert result.success is True
        assert result.error is None

    def test_api_result_error(self):
        """Test creating an APIResult with error."""
        result = APIResult(success=False, error="Something went wrong")
        assert result.success is False
        assert result.error == "Something went wrong"


class TestMetricsResult:
    """Tests for MetricsResult dataclass."""

    def test_metrics_result_default(self):
        """Test MetricsResult with default values."""
        metrics = MetricsResult()
        assert metrics.functional_correctness_score == 0.0
        assert metrics.success_rate == 0.0
        assert metrics.exact_match_rate == 0.0
        assert metrics.total_cases == 0
        assert metrics.valid_cases == 0
        assert metrics.successful_cases == 0
        assert metrics.exact_matches == 0
        assert metrics.failure_count == 0
        assert metrics.score_distribution == {}

    def test_metrics_result_from_dict(self):
        """Test creating MetricsResult from dictionary."""
        data = {
            "functional_correctness_score": 4.5,
            "success_rate": 90.0,
            "exact_match_rate": 75.0,
            "total_cases": 100,
            "valid_cases": 95,
            "successful_cases": 90,
            "exact_matches": 75,
            "failure_count": 5,
            "score_distribution": {"5": 50, "4": 40},
        }

        metrics = MetricsResult.from_dict(data)
        assert metrics.functional_correctness_score == 4.5
        assert metrics.success_rate == 90.0
        assert metrics.exact_match_rate == 75.0
        assert metrics.total_cases == 100
        assert metrics.score_distribution == {"5": 50, "4": 40}

    def test_metrics_result_to_dict(self):
        """Test converting MetricsResult to dictionary."""
        metrics = MetricsResult(
            functional_correctness_score=4.5,
            success_rate=90.0,
            total_cases=100,
        )

        result_dict = metrics.to_dict()
        assert result_dict["functional_correctness_score"] == 4.5
        assert result_dict["success_rate"] == 90.0
        assert result_dict["total_cases"] == 100

    def test_metrics_result_from_dict_partial(self):
        """Test creating MetricsResult from partial dictionary."""
        data = {"functional_correctness_score": 3.5, "total_cases": 50}

        metrics = MetricsResult.from_dict(data)
        assert metrics.functional_correctness_score == 3.5
        assert metrics.total_cases == 50
        assert metrics.success_rate == 0.0  # Default value


class TestMaterialSearchResult:
    """Tests for MaterialSearchResult dataclass."""

    def test_material_search_result(self):
        """Test creating MaterialSearchResult."""
        matches = [
            {"source_path": "/path/to/material1.mdl", "s3_path": "s3://bucket/mat1"},
            {"source_path": "/path/to/material2.mdl", "s3_path": "s3://bucket/mat2"},
        ]

        result = MaterialSearchResult(material="steel", matches=matches, match_count=2)

        assert result.material == "steel"
        assert result.match_count == 2
        assert len(result.matches) == 2


class TestAssignmentStats:
    """Tests for AssignmentStats dataclass."""

    def test_assignment_stats_default(self):
        """Test AssignmentStats with default values."""
        stats = AssignmentStats()
        assert stats.materials_created == 0
        assert stats.materials_applied == 0
        assert stats.total_prims == 0
        assert stats.failed == 0
        assert stats.bound_prim_ids == []
        assert stats.unbound_prim_ids == []

    def test_assignment_stats_with_values(self):
        """Test AssignmentStats with custom values."""
        stats = AssignmentStats(
            materials_created=10,
            materials_applied=8,
            total_prims=15,
            failed=2,
            bound_prim_ids=["/Root/A"],
            unbound_prim_ids=["/Root/B"],
        )
        assert stats.materials_created == 10
        assert stats.materials_applied == 8
        assert stats.total_prims == 15
        assert stats.failed == 2
        assert stats.bound_prim_ids == ["/Root/A"]
        assert stats.unbound_prim_ids == ["/Root/B"]


class TestDownloadStats:
    """Tests for DownloadStats dataclass."""

    def test_download_stats_default(self):
        """Test DownloadStats with default values."""
        stats = DownloadStats()
        assert stats.found_local == 0
        assert stats.downloaded == 0
        assert stats.failed == 0
        assert stats.skipped == 0

    def test_download_stats_with_values(self):
        """Test DownloadStats with custom values."""
        stats = DownloadStats(found_local=5, downloaded=3, failed=1, skipped=2)
        assert stats.found_local == 5
        assert stats.downloaded == 3
        assert stats.failed == 1
        assert stats.skipped == 2
