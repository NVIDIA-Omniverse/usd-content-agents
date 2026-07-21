# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Material Agent Build Dataset APIs."""

import asyncio
import errno
import logging
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from material_agent.api.build_dataset import (
    BuildDatasetPdfVectorstoreInput,
    BuildDatasetPdfVectorstoreOutput,
    BuildDatasetPrepareDatasetInput,
    BuildDatasetPrepareDatasetOutput,
    BuildDatasetUsdInput,
    BuildDatasetUsdOutput,
    abuild_dataset_usd,
    build_dataset_pdf_vectorstore,
    build_dataset_prepare_dataset,
    build_dataset_usd,
)


@pytest.mark.parametrize(
    "input_type",
    [
        BuildDatasetUsdInput,
        BuildDatasetPdfVectorstoreInput,
        BuildDatasetPrepareDatasetInput,
    ],
)
def test_build_dataset_inputs_project_config_inspection_oserror(
    input_type: type,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = "never-return-config-inspection-path-713"
    config_path = Path(f"cache/user:{sentinel}@assets.example.test/config.yaml")

    def raise_name_too_long(path: Path) -> bool:
        raise OSError(errno.ENAMETOOLONG, "File name too long", str(path))

    monkeypatch.setattr(Path, "exists", raise_name_too_long)

    with pytest.raises(OSError) as exc_info:
        input_type(config=config_path)

    assert exc_info.value.errno == errno.ENAMETOOLONG
    assert exc_info.value.filename == "<redacted>"
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert sentinel not in repr(exc_info.value)


# ============================================================================
# USD Dataset Building Tests
# ============================================================================


class TestBuildDatasetUsdInput:
    """Tests for BuildDatasetUsdInput validation."""

    def test_usd_input_valid(self, tmp_path):
        """Test creating valid BuildDatasetUsdInput."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")

        params = BuildDatasetUsdInput(
            config=config_file,
            extract_metadata=True,
            verbose=True,
        )

        assert params.config == config_file
        assert params.extract_metadata is True

    def test_usd_input_missing_config(self, tmp_path):
        """Test BuildDatasetUsdInput raises error for missing config."""
        sentinel = "never-log-missing-config-path-713"
        config_file = tmp_path / sentinel / "missing.yaml"

        with pytest.raises(FileNotFoundError, match="^Config file not found$") as exc:
            BuildDatasetUsdInput(config=config_file)

        assert sentinel not in str(exc.value)

    def test_usd_input_rejects_empty_dict_config(self):
        with pytest.raises(ValueError, match="Config dictionary cannot be empty"):
            BuildDatasetUsdInput(config={})

    def test_usd_input_coerces_override_paths(self, tmp_path):
        params = BuildDatasetUsdInput(
            config={"usd_path": "scene.usd"},
            source_override=str(tmp_path / "source.usd"),
            output_dir_override=str(tmp_path / "dataset"),
        )

        assert params.source_override == tmp_path / "source.usd"
        assert params.output_dir_override == tmp_path / "dataset"


class TestBuildDatasetUsd:
    """Tests for build_dataset_usd function."""

    @patch("material_agent.workflows.create_usd_data_preparation_workflow_from_config")
    def test_build_usd_single_file(self, mock_create_workflow, tmp_path):
        """Test USD dataset building for single file."""
        # Setup
        config_file = tmp_path / "config.yaml"
        config_file.write_text("usd_path: /path/to/model.usd")

        # Mock workflow
        mock_workflow = Mock()
        mock_workflow.arun = AsyncMock(
            return_value={
                "dataset_path": str(tmp_path / "dataset.jsonl"),
                "num_prims": 50,
                "num_images": 150,
            }
        )
        mock_create_workflow.return_value = mock_workflow

        # Execute
        params = BuildDatasetUsdInput(config=config_file)
        result = build_dataset_usd(params)

        # Verify
        assert result.success is True
        assert result.dataset_path == tmp_path / "dataset.jsonl"
        assert result.num_prims == 50
        assert result.num_images == 150

    @patch("material_agent.batch_processor.process_usd_batch", new_callable=AsyncMock)
    @patch("material_agent.workflows.create_usd_data_preparation_workflow_from_config")
    def test_build_usd_batch(self, mock_create_workflow, mock_batch, tmp_path):
        """Test USD dataset building in batch mode."""
        # Setup
        config_file = tmp_path / "config.yaml"
        usd_dir = tmp_path / "usd_files"
        usd_dir.mkdir()
        config_file.write_text(f"usd_dir: {usd_dir}")

        # Mock workflow
        mock_workflow = Mock()
        mock_create_workflow.return_value = mock_workflow

        # Mock batch processor (async)
        mock_batch.return_value = {
            "results": {
                "model1.usd": {
                    "status": "success",
                    "num_prims": 30,
                    "num_images": 90,
                    "output_dir": str(tmp_path / "output/model1"),
                },
                "model2.usd": {
                    "status": "success",
                    "num_prims": 40,
                    "num_images": 120,
                    "output_dir": str(tmp_path / "output/model2"),
                },
            },
            "num_files_processed": 2,
            "num_files_failed": 0,
        }

        # Execute
        params = BuildDatasetUsdInput(config=config_file)
        result = build_dataset_usd(params)

        # Verify
        assert result.success is True
        assert result.batch_results is not None
        assert len(result.batch_results) == 2

    @pytest.mark.parametrize("use_async_api", [False, True], ids=["sync", "async"])
    @patch("material_agent.workflows.create_usd_data_preparation_workflow_from_config")
    def test_build_usd_exception_diagnostics_do_not_replay_paths_or_exception(
        self,
        mock_create_workflow,
        tmp_path,
        caplog,
        use_async_api,
    ):
        """Raw paths reach the workflow but never its API diagnostics."""
        sentinels = (
            "never-log-config-path-713",
            "never-log-source-path-713",
            "never-log-output-path-713",
        )
        config_dir = tmp_path / sentinels[0]
        config_dir.mkdir()
        config_file = config_dir / "config.yaml"
        config_file.write_text("usd_path: /path/to/model.usd")
        source_dir = tmp_path / sentinels[1]
        source_dir.mkdir()
        source_path = source_dir / "scene.usd"
        source_path.write_text("#usda", encoding="utf-8")
        output_dir = tmp_path / sentinels[2]

        mock_workflow = Mock()
        mock_workflow.arun = AsyncMock(
            side_effect=RuntimeError("backend reflected " + " ".join(sentinels))
        )
        mock_create_workflow.return_value = mock_workflow

        params = BuildDatasetUsdInput(
            config=config_file,
            source_override=source_path,
            output_dir_override=output_dir,
        )
        with caplog.at_level(logging.INFO):
            if use_async_api:
                result = asyncio.run(abuild_dataset_usd(params))
            else:
                result = build_dataset_usd(params)

        assert result.success is False
        assert result.error == "USD dataset build failed"
        mock_workflow.arun.assert_awaited_once_with(
            {
                "config_path": config_file,
                "source_override": source_path,
                "output_dir_override": output_dir,
            }
        )
        observable = f"{caplog.text}\n{result.error}"
        for sentinel in sentinels:
            assert sentinel not in observable

    @patch("material_agent.workflows.create_usd_data_preparation_workflow_from_config")
    def test_build_usd_single_dict_config_with_overrides(
        self, mock_create_workflow, tmp_path
    ):
        """Test single USD dataset building with in-memory config and overrides."""
        source_path = tmp_path / "scene.usd"
        output_dir = tmp_path / "dataset"
        sentinel = "api_key=usd-public-result-secret-713"
        runtime_result = {
            "dataset_path": str(tmp_path / "dataset.jsonl"),
            "num_prims": 2,
            "num_images": 3,
            "config_dict": {"vlm": {"api_key": sentinel}},
            "details": {"api_key": sentinel},
            "runtime_collaborator": object(),
        }

        mock_workflow = Mock()
        mock_workflow.arun = AsyncMock(return_value=runtime_result)
        mock_create_workflow.return_value = mock_workflow

        config = {
            "usd_path": str(source_path),
            "vlm": {"api_key": sentinel},
        }
        params = BuildDatasetUsdInput(
            config=config,
            source_override=source_path,
            output_dir_override=output_dir,
            extract_metadata=True,
        )
        result = build_dataset_usd(params)

        assert result.success is True
        assert result.dataset_path == tmp_path / "dataset.jsonl"
        assert result.num_prims == 2
        assert result.num_images == 3
        assert result.raw_result is not runtime_result
        assert result.raw_result is not None
        assert "config_dict" not in result.raw_result
        assert "runtime_collaborator" not in result.raw_result
        assert sentinel not in repr(result)
        assert runtime_result["config_dict"]["vlm"]["api_key"] == sentinel
        mock_workflow.arun.assert_awaited_once_with(
            {
                "config_dict": config,
                "source_override": source_path,
                "output_dir_override": output_dir,
                "extract_prim_metadata": True,
            }
        )

    def test_build_usd_config_without_source_returns_error(self):
        params = BuildDatasetUsdInput(config={"not_usd": "value"})

        result = build_dataset_usd(params)

        assert result.success is False
        assert result.error == "USD dataset build failed"

    @patch("material_agent.batch_processor.process_usd_batch", new_callable=AsyncMock)
    @patch("material_agent.workflows.create_usd_data_preparation_workflow_from_config")
    def test_build_usd_batch_from_source_override_with_failures(
        self, mock_create_workflow, mock_batch, tmp_path
    ):
        """Test batch mode selected by a source directory override."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("usd_path: ignored.usd")
        usd_dir = tmp_path / "usd_files"
        usd_dir.mkdir()
        output_dir = tmp_path / "batch-output"

        mock_workflow = Mock()
        mock_create_workflow.return_value = mock_workflow
        sentinel = "api_key=batch-partial-result-secret-713"
        mock_batch.return_value = {
            "results": {
                "bad.usd": {
                    "status": "failed",
                    "api_key": sentinel,
                    "runtime_collaborator": object(),
                }
            },
            "num_files_processed": 0,
            "num_files_failed": 1,
            "config_dict": {"vlm": {"api_key": sentinel}},
        }

        params = BuildDatasetUsdInput(
            config=config_file,
            source_override=usd_dir,
            output_dir_override=output_dir,
            extract_metadata=True,
        )
        result = build_dataset_usd(params)

        assert result.success is False
        assert result.batch_results is not None
        assert result.batch_results["bad.usd"]["status"] == "failed"
        assert sentinel not in repr(result)
        assert result.raw_result is not None
        assert "config_dict" not in result.raw_result
        assert mock_batch.return_value["results"]["bad.usd"]["api_key"] == sentinel
        assert mock_batch.await_args.kwargs["usd_dir"] == usd_dir
        assert mock_batch.await_args.kwargs["batch_output_dir"] == output_dir
        assert mock_batch.await_args.kwargs["base_context"] == {
            "config_path": config_file,
            "extract_prim_metadata": True,
        }

    @patch("material_agent.batch_processor.process_usd_batch", new_callable=AsyncMock)
    @patch("material_agent.workflows.create_usd_data_preparation_workflow_from_config")
    def test_build_usd_batch_dict_config_uses_config_output_dir(
        self, mock_create_workflow, mock_batch, tmp_path
    ):
        """Test dict config batch mode uses its configured USD and output dirs."""
        usd_dir = tmp_path / "usd_files"
        usd_dir.mkdir()
        mock_create_workflow.return_value = Mock()
        mock_batch.return_value = {
            "results": {},
            "num_files_processed": 0,
            "num_files_failed": 0,
        }

        params = BuildDatasetUsdInput(
            config={"usd_dir": str(usd_dir), "output_dir": "configured-output"}
        )
        result = build_dataset_usd(params)

        assert result.success is True
        assert mock_batch.await_args.kwargs["usd_dir"] == usd_dir
        assert mock_batch.await_args.kwargs["batch_output_dir"] == Path(
            "configured-output"
        )
        assert mock_batch.await_args.kwargs["base_context"] == {
            "config_dict": {"usd_dir": str(usd_dir), "output_dir": "configured-output"}
        }

    @patch("material_agent.batch_processor.process_usd_batch", new_callable=AsyncMock)
    @patch("material_agent.workflows.create_usd_data_preparation_workflow_from_config")
    def test_build_usd_batch_file_config_resolves_relative_output_dir(
        self, mock_create_workflow, mock_batch, tmp_path
    ):
        usd_dir = tmp_path / "usd_files"
        usd_dir.mkdir()
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            f"usd_dir: {usd_dir}\noutput_dir: relative-output\n",
            encoding="utf-8",
        )
        mock_create_workflow.return_value = Mock()
        mock_batch.return_value = {
            "results": {},
            "num_files_processed": 0,
            "num_files_failed": 0,
        }

        result = build_dataset_usd(BuildDatasetUsdInput(config=config_file))

        assert result.success is True
        assert (
            mock_batch.await_args.kwargs["batch_output_dir"]
            == (tmp_path / "relative-output").resolve()
        )


# ============================================================================
# PDF VectorStore Building Tests
# ============================================================================


class TestBuildDatasetPdfVectorstoreInput:
    """Tests for BuildDatasetPdfVectorstoreInput validation."""

    def test_pdf_input_valid(self, tmp_path):
        """Test creating valid BuildDatasetPdfVectorstoreInput."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")

        params = BuildDatasetPdfVectorstoreInput(
            config=config_file,
            verbose=True,
        )

        assert params.config == config_file
        assert params.verbose is True

    def test_pdf_input_rejects_empty_dict_config(self):
        with pytest.raises(ValueError, match="Config dictionary cannot be empty"):
            BuildDatasetPdfVectorstoreInput(config={})

    def test_pdf_input_missing_config_does_not_replay_path(self, tmp_path):
        sentinel = "api_key=missing-pdf-config-713"
        config_file = tmp_path / sentinel / "missing.yaml"

        with pytest.raises(FileNotFoundError, match="^Config file not found$") as exc:
            BuildDatasetPdfVectorstoreInput(config=config_file)

        assert sentinel not in str(exc.value)

    def test_pdf_input_coerces_override_paths(self, tmp_path):
        params = BuildDatasetPdfVectorstoreInput(
            config={"pdf_path": "doc.pdf"},
            source_override=str(tmp_path / "doc.pdf"),
            output_dir_override=str(tmp_path / "vectorstore"),
        )

        assert params.source_override == tmp_path / "doc.pdf"
        assert params.output_dir_override == tmp_path / "vectorstore"


class TestBuildDatasetPdfVectorstore:
    """Tests for build_dataset_pdf_vectorstore function."""

    @patch(
        "material_agent.workflows.factory.create_pdf_vectorstore_workflow_from_config"
    )
    def test_build_pdf_vectorstore_success(self, mock_create_workflow, tmp_path):
        """Test PDF vectorstore building success."""
        # Setup
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")

        # Mock workflow
        mock_workflow = Mock()
        mock_workflow.arun = AsyncMock(
            return_value={
                "workflow_completed": True,
                "vectorstore_result": {
                    "save_path": str(tmp_path / "vectorstore"),
                    "num_documents_indexed": 100,
                    "num_texts": 80,
                    "num_images": 20,
                    "embedding_dimension": 768,
                },
                "extraction_result": {"document_count": 10},
                "split_result": {"total_files_created": 100},
            }
        )
        mock_create_workflow.return_value = mock_workflow

        # Execute
        params = BuildDatasetPdfVectorstoreInput(config=config_file)
        result = build_dataset_pdf_vectorstore(params)

        # Verify
        assert result.success is True
        assert result.vectorstore_path == tmp_path / "vectorstore"
        assert result.num_documents_indexed == 100
        assert result.num_texts == 80
        assert result.num_images == 20

    @patch(
        "material_agent.workflows.factory.create_pdf_vectorstore_workflow_from_config"
    )
    def test_build_pdf_vectorstore_failed(self, mock_create_workflow, tmp_path, caplog):
        """Test PDF vectorstore building when workflow fails."""
        # Setup
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")

        # Mock workflow that fails
        mock_workflow = Mock()
        sentinel = "api_key=pdf-partial-result-secret-713"
        mock_workflow.arun = AsyncMock(
            return_value={
                "workflow_completed": False,
                "error": sentinel,
                "config_dict": {"vlm": {"api_key": sentinel}},
                "runtime_collaborator": object(),
            }
        )
        mock_create_workflow.return_value = mock_workflow

        # Execute
        params = BuildDatasetPdfVectorstoreInput(config=config_file)
        with caplog.at_level(logging.INFO):
            result = build_dataset_pdf_vectorstore(params)

        # Verify
        assert result.success is False
        assert result.error == "PDF vectorstore build failed"
        assert result.raw_result == {
            "workflow_completed": False,
            "error": "PDF vectorstore build failed",
        }
        assert sentinel not in caplog.text
        assert sentinel not in repr(result)

    @patch(
        "material_agent.workflows.factory.create_pdf_vectorstore_workflow_from_config"
    )
    def test_build_pdf_vectorstore_exception_diagnostics_are_secret_safe(
        self, mock_create_workflow, tmp_path, caplog
    ):
        sentinels = (
            "api_key=pdf-config-secret-713",
            "api_key=pdf-source-secret-713",
            "api_key=pdf-output-secret-713",
            "opaque-pdf-exception-secret-713",
        )
        config_dir = tmp_path / sentinels[0]
        config_dir.mkdir()
        config_file = config_dir / "config.yaml"
        config_file.write_text("# test config", encoding="utf-8")
        source_path = tmp_path / sentinels[1] / "doc.pdf"
        output_dir = tmp_path / sentinels[2]

        mock_workflow = Mock()
        mock_workflow.arun = AsyncMock(side_effect=RuntimeError(sentinels[3]))
        mock_create_workflow.return_value = mock_workflow

        params = BuildDatasetPdfVectorstoreInput(
            config=config_file,
            source_override=source_path,
            output_dir_override=output_dir,
        )
        with caplog.at_level(logging.INFO):
            result = build_dataset_pdf_vectorstore(params)

        assert result.success is False
        assert result.error == "PDF vectorstore build failed"
        mock_workflow.arun.assert_awaited_once_with(
            initial_context={
                "source_override": str(source_path),
                "output_dir_override": str(output_dir),
                "verbose": False,
                "config_path": str(config_file),
            }
        )
        observable = f"{caplog.text}\n{result.error}"
        for sentinel in sentinels:
            assert sentinel not in observable
        assert all(
            record.exc_info is None
            for record in caplog.records
            if record.name == "material_agent.api.build_dataset"
        )

    @patch(
        "material_agent.workflows.factory.create_pdf_vectorstore_workflow_from_config"
    )
    def test_build_pdf_vectorstore_dict_config_with_overrides(
        self, mock_create_workflow, tmp_path
    ):
        """Test PDF vectorstore context for dict config and path overrides."""
        source_path = tmp_path / "docs"
        output_dir = tmp_path / "vectorstore"
        sentinel = "api_key=pdf-public-result-secret-713"
        runtime_result = {
            "workflow_completed": True,
            "vectorstore_result": {
                "num_documents_indexed": 1,
                "api_key": sentinel,
            },
            "extraction_result": {
                "document_count": 1,
                "config_dict": {"vlm": {"api_key": sentinel}},
            },
            "split_result": {
                "total_files_created": 1,
                "runtime_collaborator": object(),
            },
            "config_dict": {"vlm": {"api_key": sentinel}},
        }

        mock_workflow = Mock()
        mock_workflow.arun = AsyncMock(return_value=runtime_result)
        mock_create_workflow.return_value = mock_workflow

        config = {"pdf_dir": str(source_path), "vlm": {"api_key": sentinel}}
        params = BuildDatasetPdfVectorstoreInput(
            config=config,
            source_override=source_path,
            output_dir_override=output_dir,
            verbose=True,
        )
        result = build_dataset_pdf_vectorstore(params)

        assert result.success is True
        assert result.vectorstore_path is None
        assert result.num_documents_indexed == 1
        assert result.extraction_result == {"document_count": 1}
        assert result.split_result == {"total_files_created": 1}
        assert result.raw_result is not runtime_result
        assert result.raw_result is not None
        assert "config_dict" not in result.raw_result
        assert sentinel not in repr(result)
        assert runtime_result["config_dict"]["vlm"]["api_key"] == sentinel
        mock_workflow.arun.assert_awaited_once_with(
            initial_context={
                "source_override": str(source_path),
                "output_dir_override": str(output_dir),
                "verbose": True,
                "config_dict": config,
            }
        )


# ============================================================================
# Prepare Dataset Tests
# ============================================================================


class TestBuildDatasetPrepareDatasetInput:
    """Tests for BuildDatasetPrepareDatasetInput validation."""

    def test_prepare_input_valid(self, tmp_path):
        """Test creating valid BuildDatasetPrepareDatasetInput."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")

        params = BuildDatasetPrepareDatasetInput(
            config=config_file,
            verbose=True,
        )

        assert params.config == config_file
        assert params.verbose is True

    def test_prepare_input_rejects_empty_dict_config(self):
        with pytest.raises(ValueError, match="Config dictionary cannot be empty"):
            BuildDatasetPrepareDatasetInput(config={})

    def test_prepare_input_missing_config_does_not_replay_path(self, tmp_path):
        sentinel = "api_key=missing-prepare-config-713"
        config_file = tmp_path / sentinel / "missing.yaml"

        with pytest.raises(FileNotFoundError, match="^Config file not found$") as exc:
            BuildDatasetPrepareDatasetInput(config=config_file)

        assert sentinel not in str(exc.value)

    def test_prepare_input_coerces_override_paths(self, tmp_path):
        params = BuildDatasetPrepareDatasetInput(
            config={"models": []},
            vector_store_override=str(tmp_path / "vectorstore"),
            dataset_override=str(tmp_path / "dataset"),
        )

        assert params.vector_store_override == tmp_path / "vectorstore"
        assert params.dataset_override == tmp_path / "dataset"


class TestBuildDatasetPrepareDataset:
    """Tests for build_dataset_prepare_dataset function."""

    @patch(
        "material_agent.workflows.factory.create_prepare_dataset_workflow_from_config"
    )
    def test_prepare_dataset_success(self, mock_create_workflow, tmp_path):
        """Test dataset preparation success."""
        # Setup
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")

        # Mock workflow
        mock_workflow = Mock()
        mock_workflow.arun = AsyncMock(
            return_value={
                "dataset_entries": [
                    {"id": "entry1", "specification": "spec1"},
                    {"id": "entry2", "specification": "spec2"},
                ],
                "failed_models": ["model3"],
                "dataset_jsonl_path": str(tmp_path / "dataset.jsonl"),
            }
        )
        mock_create_workflow.return_value = mock_workflow

        # Execute
        params = BuildDatasetPrepareDatasetInput(config=config_file)
        result = build_dataset_prepare_dataset(params)

        # Verify
        assert result.success is True
        assert len(result.dataset_entries) == 2
        assert result.failed_models == ["model3"]
        assert result.dataset_jsonl_path == tmp_path / "dataset.jsonl"

    @patch(
        "material_agent.workflows.factory.create_prepare_dataset_workflow_from_config"
    )
    def test_prepare_dataset_not_complete(self, mock_create_workflow, tmp_path):
        """Test dataset preparation when workflow doesn't complete."""
        # Setup
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")

        # Mock workflow that doesn't complete
        mock_workflow = Mock()
        sentinel = "api_key=prepare-partial-result-secret-713"
        mock_workflow.arun = AsyncMock(
            return_value={
                "dataset_entries": None,
                "error": sentinel,
                "config_dict": {"vlm": {"api_key": sentinel}},
                "runtime_collaborator": object(),
            }
        )
        mock_create_workflow.return_value = mock_workflow

        # Execute
        params = BuildDatasetPrepareDatasetInput(config=config_file)
        result = build_dataset_prepare_dataset(params)

        # Verify
        assert result.success is False
        assert "did not complete" in result.error.lower()
        assert result.raw_result == {
            "dataset_entries": None,
            "error": "Prepare dataset workflow did not complete successfully",
        }
        assert sentinel not in repr(result)

    @patch(
        "material_agent.workflows.factory.create_prepare_dataset_workflow_from_config"
    )
    def test_prepare_dataset_exception_diagnostics_are_secret_safe(
        self, mock_create_workflow, tmp_path, caplog
    ):
        sentinels = (
            "api_key=prepare-config-secret-713",
            "api_key=vector-store-secret-713",
            "api_key=dataset-secret-713",
            "opaque-prepare-exception-secret-713",
        )
        config_dir = tmp_path / sentinels[0]
        config_dir.mkdir()
        config_file = config_dir / "config.yaml"
        config_file.write_text("# test config", encoding="utf-8")
        vectorstore = tmp_path / sentinels[1]
        dataset = tmp_path / sentinels[2]

        mock_workflow = Mock()
        mock_workflow.arun = AsyncMock(side_effect=RuntimeError(sentinels[3]))
        mock_create_workflow.return_value = mock_workflow

        params = BuildDatasetPrepareDatasetInput(
            config=config_file,
            vector_store_override=vectorstore,
            dataset_override=dataset,
        )
        with caplog.at_level(logging.INFO):
            result = build_dataset_prepare_dataset(params)

        assert result.success is False
        assert result.error == "Prepare dataset failed"
        mock_workflow.arun.assert_awaited_once_with(
            initial_context={
                "vector_store_override": str(vectorstore),
                "dataset_override": str(dataset),
                "verbose": False,
                "config_path": str(config_file),
            }
        )
        observable = f"{caplog.text}\n{result.error}"
        for sentinel in sentinels:
            assert sentinel not in observable
        assert all(
            record.exc_info is None
            for record in caplog.records
            if record.name == "material_agent.api.build_dataset"
        )

    @patch(
        "material_agent.workflows.factory.create_prepare_dataset_workflow_from_config"
    )
    def test_prepare_dataset_dict_config_with_overrides(
        self, mock_create_workflow, tmp_path
    ):
        """Test prepare-dataset context for dict config and path overrides."""
        vectorstore = tmp_path / "vectorstore"
        dataset = tmp_path / "dataset"
        sentinel = "api_key=prepare-public-result-secret-713"
        runtime_result = {
            "dataset_entries": [
                {
                    "id": "safe-entry",
                    "api_key": sentinel,
                    "runtime_collaborator": object(),
                }
            ],
            "failed_models": ["safe-model"],
            "dataset_jsonl_path": None,
            "config_dict": {"vlm": {"api_key": sentinel}},
        }

        mock_workflow = Mock()
        mock_workflow.arun = AsyncMock(return_value=runtime_result)
        mock_create_workflow.return_value = mock_workflow

        config = {"model_numbers": [], "vlm": {"api_key": sentinel}}
        params = BuildDatasetPrepareDatasetInput(
            config=config,
            vector_store_override=vectorstore,
            dataset_override=dataset,
            verbose=True,
        )
        result = build_dataset_prepare_dataset(params)

        assert result.success is True
        assert result.dataset_jsonl_path is None
        assert result.failed_models == ["safe-model"]
        assert result.dataset_entries[0]["id"] == "safe-entry"
        assert sentinel not in repr(result)
        assert result.raw_result is not runtime_result
        assert result.raw_result is not None
        assert "config_dict" not in result.raw_result
        assert runtime_result["config_dict"]["vlm"]["api_key"] == sentinel
        mock_workflow.arun.assert_awaited_once_with(
            initial_context={
                "vector_store_override": str(vectorstore),
                "dataset_override": str(dataset),
                "verbose": True,
                "config_dict": config,
            }
        )


# ============================================================================
# Output Tests
# ============================================================================


class TestBuildDatasetOutputs:
    """Tests for build dataset output dataclasses."""

    def test_usd_output_success(self, tmp_path):
        """Test creating successful BuildDatasetUsdOutput."""
        output = BuildDatasetUsdOutput(
            success=True,
            dataset_path=tmp_path / "dataset.jsonl",
            num_prims=100,
            num_images=300,
        )

        assert output.success is True
        assert output.dataset_path == tmp_path / "dataset.jsonl"
        assert output.num_prims == 100

    def test_pdf_output_success(self, tmp_path):
        """Test creating successful BuildDatasetPdfVectorstoreOutput."""
        output = BuildDatasetPdfVectorstoreOutput(
            success=True,
            vectorstore_path=tmp_path / "vectorstore",
            num_documents_indexed=50,
            num_texts=40,
            num_images=10,
        )

        assert output.success is True
        assert output.vectorstore_path == tmp_path / "vectorstore"
        assert output.num_documents_indexed == 50

    def test_prepare_output_success(self, tmp_path):
        """Test creating successful BuildDatasetPrepareDatasetOutput."""
        output = BuildDatasetPrepareDatasetOutput(
            success=True,
            dataset_jsonl_path=tmp_path / "dataset.jsonl",
            dataset_entries=[{"id": "1"}, {"id": "2"}],
            failed_models=["model3"],
        )

        assert output.success is True
        assert len(output.dataset_entries) == 2
        assert output.failed_models == ["model3"]
