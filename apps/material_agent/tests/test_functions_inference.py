# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Material Agent inference functions."""

import logging
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from PIL import Image

from material_agent.functions.inference import (
    assign_material,
    async_batch_assign_materials,
    batch_assign_materials,
    reconcile_untrusted_spec_evidence,
)
from material_agent.materials import UNKNOWN_MATERIAL_SENTINEL


@pytest.mark.parametrize(
    ("visual_material", "spec_material", "status", "review_required"),
    [
        ("Steel", "304 Stainless Steel", "corroborated", False),
        ("Steel", "Brass", "conflict", True),
        (UNKNOWN_MATERIAL_SENTINEL, "Brass", "unsupported_spec_material", True),
    ],
)
def test_untrusted_spec_only_corroborates_image_supported_material(
    visual_material: str,
    spec_material: str,
    status: str,
    review_required: bool,
) -> None:
    visual_prediction = {"material": visual_material, "confidence": 0.9}
    entry = {
        "untrusted_spec_evidence": {
            "extracted_text": f"Part: Housing\n- Material Type: {spec_material}"
        }
    }

    result = reconcile_untrusted_spec_evidence(visual_prediction, entry)

    assert result["material"] == visual_material
    assert result["confidence"] == 0.9
    reconciliation = result["evidence_reconciliation"]
    assert reconciliation["status"] == status
    assert reconciliation["review_required"] is review_required
    assert reconciliation["visual_material"] == visual_material
    assert reconciliation["untrusted_spec_material_claims"] == [spec_material]
    if status == "corroborated":
        assert reconciliation["corroborating_spec_materials"] == [spec_material]
        assert "conflicting_spec_materials" not in reconciliation
    else:
        assert reconciliation["conflicting_spec_materials"] == [spec_material]
        assert spec_material != result["material"]


def test_image_only_prediction_is_unchanged_without_spec_claims() -> None:
    visual_prediction = {"material": "Steel", "confidence": 0.9}

    result = reconcile_untrusted_spec_evidence(visual_prediction, {})

    assert result is visual_prediction
    assert result == {"material": "Steel", "confidence": 0.9}


class TestAssignMaterial:
    """Tests for the assign_material function."""

    def test_assign_material_with_file_paths(
        self, mock_vlm, mock_llm, sample_image_files
    ):
        """Test assign_material with file path inputs."""
        result = assign_material(
            vlm=mock_vlm,
            text="This is a car wheel. Materials: steel, rubber, plastic",
            images=sample_image_files,
            llm=mock_llm,
        )

        # Verify VLM was called with correct parameters
        mock_vlm.generate.assert_called_once()
        call_kwargs = mock_vlm.generate.call_args.kwargs
        assert call_kwargs["images"] == sample_image_files
        assert "car wheel" in call_kwargs["prompt"]

        # Verify LLM was called for parsing
        mock_llm.invoke.assert_called_once()

        # Verify result structure
        assert isinstance(result, dict)
        assert "material" in result
        assert "original_response" in result
        assert result["material"] == "matt black rubber"

    def test_assign_material_with_pil_images(
        self, mock_vlm, mock_llm, sample_pil_images
    ):
        """Test assign_material with PIL Image inputs."""
        result = assign_material(
            vlm=mock_vlm,
            text="This is a car door. Materials: steel, glass, plastic",
            images=sample_pil_images,
            llm=mock_llm,
        )

        # Verify VLM was called with PIL Images
        mock_vlm.generate.assert_called_once()
        call_kwargs = mock_vlm.generate.call_args.kwargs
        assert call_kwargs["images"] == sample_pil_images
        assert all(isinstance(img, Image.Image) for img in call_kwargs["images"])

        # Verify result
        assert result["material"] == "matt black rubber"
        assert "original_response" in result

    def test_assign_material_with_mixed_images(
        self, mock_vlm, mock_llm, sample_image_files, sample_pil_images
    ):
        """Test assign_material with mixed file paths and PIL Images."""
        mixed_images = [
            sample_image_files[0],
            sample_pil_images[0],
            sample_image_files[1],
        ]

        result = assign_material(
            vlm=mock_vlm,
            text="This is a car seat. Materials: leather, fabric, plastic",
            images=mixed_images,
            llm=mock_llm,
        )

        # Verify VLM was called with mixed image types
        mock_vlm.generate.assert_called_once()
        call_kwargs = mock_vlm.generate.call_args.kwargs
        assert len(call_kwargs["images"]) == 3

        # Verify result
        assert isinstance(result, dict)
        assert "material" in result

    def test_assign_material_with_custom_prompt(
        self, mock_vlm, mock_llm, sample_pil_images
    ):
        """Test assign_material with custom system prompt and invoke_kwargs."""
        custom_prompt = "You are a materials science expert."

        result = assign_material(
            vlm=mock_vlm,
            text="Identify the material",
            images=sample_pil_images,
            llm=mock_llm,
            system_prompt=custom_prompt,
            invoke_kwargs={"temperature": 0.5, "max_tokens": 512},
        )

        # Verify custom parameters were passed
        call_kwargs = mock_vlm.generate.call_args.kwargs
        assert call_kwargs["system_prompt"] == custom_prompt
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 512

        assert isinstance(result, dict)

    def test_assign_material_with_llm_parse_failure(
        self, mock_vlm, mock_llm_with_invalid_json, sample_pil_images
    ):
        """Test assign_material when LLM returns invalid JSON."""
        result = assign_material(
            vlm=mock_vlm,
            text="This is a test object. Materials: material1, material2",
            images=sample_pil_images,
            llm=mock_llm_with_invalid_json,
        )

        # Should return fallback structure
        assert isinstance(result, dict)
        assert result["material"] == "Unable to parse"
        assert "original_response" in result
        assert "Looking at the images" in result["original_response"]

    def test_assign_material_with_llm_exception(
        self, mock_vlm, sample_pil_images, caplog
    ):
        """Test assign_material when LLM raises an exception."""
        mock_llm_error = Mock()
        mock_llm_error.invoke = MagicMock(side_effect=Exception("LLM error"))

        with caplog.at_level(logging.ERROR):
            result = assign_material(
                vlm=mock_vlm,
                text="Test materials",
                images=sample_pil_images,
                llm=mock_llm_error,
            )

        # Should return error fallback structure
        assert result["material"] == "Error during parsing"
        assert "original_response" in result
        assert "Error parsing VLM response with LLM" in caplog.text

    def test_assign_material_preserves_unknown_sentinel_without_llm(
        self, mock_vlm, mock_llm, sample_pil_images
    ):
        """Material assignment should preserve explicit unknown sentinel output."""
        mock_vlm.generate.return_value = "__UNKNOWN__"

        result = assign_material(
            vlm=mock_vlm,
            text="Test materials",
            images=sample_pil_images,
            llm=mock_llm,
        )

        assert result["material"] == "__UNKNOWN__"
        assert "__UNKNOWN__" in result["original_response"]
        mock_llm.invoke.assert_not_called()


class TestBatchAssignMaterials:
    """Tests for the batch_assign_materials function."""

    def test_batch_assign_materials_basic(
        self,
        mock_vlm_with_varied_responses,
        mock_llm_with_varied_responses,
        sample_entries,
        tmp_path,
    ):
        """Test basic batch processing with mixed image types."""
        results = batch_assign_materials(
            vlm=mock_vlm_with_varied_responses,
            entries=sample_entries,
            llm=mock_llm_with_varied_responses,
            image_base_dir=tmp_path,
        )

        # Verify results structure
        assert len(results) == 3
        assert all("id" in r for r in results)
        assert all("vlm_response" in r for r in results)
        assert all("status" in r for r in results)

        # Verify successful processing
        assert results[0]["status"] == "success"
        assert results[0]["vlm_response"]["material"] == "rubber"

        assert results[1]["status"] == "success"
        assert results[1]["vlm_response"]["material"] == "steel"

        assert results[2]["status"] == "success"
        assert results[2]["vlm_response"]["material"] == "leather"

    def test_batch_assign_materials_with_callbacks(
        self, mock_vlm, mock_llm, sample_entries, tmp_path
    ):
        """Test batch processing with progress and error callbacks."""
        progress_calls = []
        error_calls = []

        def on_progress(entry_id, response):
            progress_calls.append((entry_id, response))

        def on_error(entry_id, error):
            error_calls.append((entry_id, error))

        batch_assign_materials(
            vlm=mock_vlm,
            entries=sample_entries,
            llm=mock_llm,
            image_base_dir=tmp_path,
            on_progress=on_progress,
            on_error=on_error,
        )

        # Verify callbacks were called
        assert len(progress_calls) == 3
        assert all(
            call[0] in ["entry_001", "entry_002", "entry_003"]
            for call in progress_calls
        )
        assert len(error_calls) == 0  # No errors expected

    def test_batch_assign_materials_with_missing_images(self, mock_vlm, mock_llm):
        """Test batch processing with missing image files."""
        entries = [
            {
                "id": "missing_001",
                "text": "Test material",
                "images": ["nonexistent1.png", "nonexistent2.png"],
            }
        ]

        error_calls = []

        def on_error(entry_id, error):
            error_calls.append((entry_id, error))

        results = batch_assign_materials(
            vlm=mock_vlm,
            entries=entries,
            llm=mock_llm,
            on_error=on_error,
        )

        # Verify error handling
        assert len(results) == 1
        assert results[0]["status"] == "error"
        assert "Missing images" in results[0]["error"]
        assert len(error_calls) == 1
        assert "missing_001" in error_calls[0][0]

    def test_batch_assign_materials_with_vlm_error(
        self, mock_vlm_with_error, mock_llm, sample_entries, tmp_path
    ):
        """Test batch processing when VLM raises an error."""
        results = batch_assign_materials(
            vlm=mock_vlm_with_error,
            entries=sample_entries[:1],  # Use only first entry
            llm=mock_llm,
            image_base_dir=tmp_path,
        )

        # Verify error handling
        assert len(results) == 1
        assert results[0]["status"] == "error"
        assert "VLM inference failed" in results[0]["error"]

    def test_batch_assign_materials_with_invalid_image_type(self, mock_vlm, mock_llm):
        """Test batch processing with invalid image types."""
        entries = [
            {
                "id": "invalid_001",
                "text": "Test material",
                "images": [123, "not_an_image"],  # Invalid types
            }
        ]

        results = batch_assign_materials(
            vlm=mock_vlm,
            entries=entries,
            llm=mock_llm,
        )

        # Should create exactly one error result with consolidated error message
        assert len(results) == 1
        assert results[0]["status"] == "error"
        assert results[0]["id"] == "invalid_001"

        # The error message should contain both issues
        error_msg = results[0]["error"]
        assert "Unsupported image type" in error_msg
        assert "Missing images" in error_msg

    def test_batch_assign_materials_empty_entries(self, mock_vlm, mock_llm):
        """Test batch processing with empty entries list."""
        results = batch_assign_materials(
            vlm=mock_vlm,
            entries=[],
            llm=mock_llm,
        )

        assert len(results) == 0

    def test_batch_assign_materials_with_absolute_paths(
        self, mock_vlm, mock_llm, sample_image_files
    ):
        """Test batch processing with absolute file paths."""
        entries = [
            {
                "id": "abs_001",
                "text": "Test material",
                "images": [str(p.absolute()) for p in sample_image_files],
            }
        ]

        results = batch_assign_materials(
            vlm=mock_vlm,
            entries=entries,
            llm=mock_llm,
            image_base_dir=Path(
                "/some/other/dir"
            ),  # Should be ignored for absolute paths
        )

        assert len(results) == 1
        assert results[0]["status"] == "success"

    def test_batch_assign_materials_logging(
        self, mock_vlm, mock_llm, sample_entries, tmp_path, caplog
    ):
        """Test that batch processing logs appropriate messages."""
        with caplog.at_level(logging.INFO):
            batch_assign_materials(
                vlm=mock_vlm,
                entries=sample_entries,
                llm=mock_llm,
                image_base_dir=tmp_path,
            )

        # Check for expected log messages
        assert "Starting batch material assignment for 3 entries" in caplog.text
        assert "Sequential processing complete" in caplog.text
        assert "3 successful, 0 failed" in caplog.text

    def test_batch_assign_materials_custom_parameters(
        self, mock_vlm, mock_llm, sample_entries, tmp_path
    ):
        """Test batch processing with custom temperature and max_tokens via invoke_kwargs."""
        custom_prompt = "Custom system prompt for testing"

        batch_assign_materials(
            vlm=mock_vlm,
            entries=sample_entries[:1],
            llm=mock_llm,
            image_base_dir=tmp_path,
            system_prompt=custom_prompt,
            invoke_kwargs={"temperature": 0.8, "max_tokens": 1024},
        )

        # Verify custom parameters were passed to VLM
        call_kwargs = mock_vlm.generate.call_args.kwargs
        assert call_kwargs["system_prompt"] == custom_prompt
        assert call_kwargs["temperature"] == 0.8
        assert call_kwargs["max_tokens"] == 1024

    @pytest.mark.asyncio
    async def test_async_batch_assign_materials_delegates_with_material_options(
        self,
        monkeypatch,
        mock_vlm,
        mock_llm,
    ):
        import material_agent.functions.inference as inference_module

        calls = []

        async def fake_async_batch_classify_objects(**kwargs):
            calls.append(kwargs)
            prediction = {"material": "Steel"}
            kwargs["on_prediction"]("entry-1", prediction)
            return [
                {
                    "id": "entry-1",
                    "status": "success",
                    "vlm_response": prediction,
                }
            ]

        monkeypatch.setattr(
            inference_module,
            "async_batch_classify_objects",
            fake_async_batch_classify_objects,
        )

        predictions = []
        results = await async_batch_assign_materials(
            vlm=mock_vlm,
            entries=[
                {
                    "id": "entry-1",
                    "text": "part",
                    "images": [],
                    "untrusted_spec_evidence": {
                        "extracted_text": "Material Type: Brass"
                    },
                }
            ],
            llm=mock_llm,
            processed_ids={"already-done"},
            on_prediction=lambda _entry_id, prediction: predictions.append(prediction),
            max_workers=2,
            max_retries=1,
        )

        assert results[0]["vlm_response"]["material"] == "Steel"
        assert predictions[0]["evidence_reconciliation"]["status"] == "conflict"
        assert calls[0]["output_key"] == "material"
        assert calls[0]["unknown_sentinel"] == UNKNOWN_MATERIAL_SENTINEL
        assert calls[0]["processed_ids"] == {"already-done"}
        assert calls[0]["max_workers"] == 2
        assert calls[0]["max_retries"] == 1


class TestIntegrationScenarios:
    """Integration tests for complex scenarios."""

    def test_mixed_batch_with_partial_failures(
        self, mock_vlm, mock_llm, sample_image_files, sample_pil_images, tmp_path
    ):
        """Test batch processing with some successful and some failed entries."""
        entries = [
            {
                "id": "success_001",
                "text": "Valid entry with images",
                "images": sample_pil_images[:2],
            },
            {
                "id": "fail_001",
                "text": "Entry with missing images",
                "images": ["missing1.png", "missing2.png"],
            },
            {
                "id": "success_002",
                "text": "Another valid entry",
                "images": [str(sample_image_files[0])],
            },
        ]

        results = batch_assign_materials(
            vlm=mock_vlm,
            entries=entries,
            llm=mock_llm,
            image_base_dir=tmp_path,
        )

        # Verify mixed results
        assert len(results) == 3
        successful = [r for r in results if r["status"] == "success"]
        failed = [r for r in results if r["status"] == "error"]

        assert len(successful) == 2
        assert len(failed) == 1
        assert failed[0]["id"] == "fail_001"

    @patch("material_agent.functions.inference.logger")
    def test_debug_logging(self, mock_logger, mock_vlm, mock_llm, sample_pil_images):
        """Test that debug logging works correctly."""
        assign_material(
            vlm=mock_vlm,
            text="Debug test",
            images=sample_pil_images,
            llm=mock_llm,
        )

        # Verify debug calls were made
        debug_calls = list(mock_logger.debug.call_args_list)
        assert len(debug_calls) > 0
        assert any("Running material assignment" in str(call) for call in debug_calls)
