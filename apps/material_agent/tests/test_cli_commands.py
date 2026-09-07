# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coverage-oriented tests for material_agent CLI commands."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

import material_agent.api as api
import material_agent.cli as cli

runner = CliRunner()


def _patch_cli_common(monkeypatch):
    logger = SimpleNamespace(
        info=lambda *args, **kwargs: None,
        debug=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(cli, "setup_logging", lambda **kwargs: logger)
    monkeypatch.setattr(
        cli,
        "TelemetryConfig",
        lambda: SimpleNamespace(enabled=False, service_name="material", exporters=[]),
    )
    monkeypatch.setattr(cli, "initialize_telemetry", lambda config: None)
    monkeypatch.setattr(cli.atexit, "register", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli.console, "print", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli.console, "print_exception", lambda *args, **kwargs: None)


def _config(tmp_path: Path, contents: str = "project:\n  name: test\n") -> Path:
    config = tmp_path / "config.yaml"
    config.write_text(contents, encoding="utf-8")
    return config


def _capture_init(monkeypatch, cls_name: str, fn_name: str, result):
    captured: dict[str, object] = {}

    class _Input:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(api, cls_name, _Input)
    monkeypatch.setattr(api, fn_name, lambda params: result)
    return captured


def test_benchmark_command_success(monkeypatch, tmp_path: Path) -> None:
    _patch_cli_common(monkeypatch)
    config = _config(tmp_path)
    dataset = tmp_path / "dataset.jsonl"
    output = tmp_path / "results"
    metrics = SimpleNamespace(
        functional_correctness_score=4.5,
        success_rate=90,
        exact_match_rate=80,
        total_cases=10,
        valid_cases=10,
        successful_cases=9,
        exact_matches=8,
        failure_count=1,
        score_distribution={5: 3, 4: 6, 0: 1},
    )
    captured = _capture_init(
        monkeypatch,
        "BenchmarkInput",
        "run_benchmark",
        SimpleNamespace(
            success=True,
            metrics=metrics,
            evaluation_path="eval.json",
            predictions_path="predictions.jsonl",
        ),
    )

    result = runner.invoke(
        cli.app,
        [
            "benchmark",
            str(config),
            "--dataset",
            str(dataset),
            "--output",
            str(output),
            "--resume",
        ],
    )

    assert result.exit_code == 0
    assert captured["config"] == config
    assert captured["dataset_override"] == dataset
    assert captured["output_dir_override"] == output
    assert captured["resume"] is True
    assert captured["stream_predictions"] is True


def test_benchmark_command_rejects_missing_config(monkeypatch, tmp_path: Path) -> None:
    _patch_cli_common(monkeypatch)

    result = runner.invoke(cli.app, ["benchmark", str(tmp_path / "missing.yaml")])

    assert result.exit_code == 1


def test_evaluate_command_success(monkeypatch, tmp_path: Path) -> None:
    _patch_cli_common(monkeypatch)
    config = _config(tmp_path)
    metrics = SimpleNamespace(
        functional_correctness_score=4.0,
        success_rate=88,
        exact_match_rate=70,
        total_cases=8,
        valid_cases=8,
        successful_cases=7,
        exact_matches=6,
        failure_count=1,
        score_distribution={5: 2, 4: 4, 3: 2},
    )
    captured = _capture_init(
        monkeypatch,
        "EvaluateInput",
        "run_evaluate",
        SimpleNamespace(
            success=True,
            metrics=metrics,
            evaluation_path="eval.json",
            html_report_path="report.html",
        ),
    )

    result = runner.invoke(cli.app, ["evaluate", str(config)])

    assert result.exit_code == 0
    assert captured["config"] == config
    assert captured["predictions_override"] is None


def test_evaluate_command_rejects_missing_predictions_override(
    monkeypatch, tmp_path: Path
) -> None:
    _patch_cli_common(monkeypatch)
    config = _config(tmp_path)

    result = runner.invoke(
        cli.app,
        ["evaluate", str(config), str(tmp_path / "missing_predictions.jsonl")],
    )

    assert result.exit_code == 1


def test_build_pdf_vectorstore_command_success(monkeypatch, tmp_path: Path) -> None:
    _patch_cli_common(monkeypatch)
    config = _config(tmp_path)
    source = tmp_path / "docs"
    output = tmp_path / "vectorstore"
    captured = _capture_init(
        monkeypatch,
        "BuildDatasetPdfVectorstoreInput",
        "build_dataset_pdf_vectorstore",
        SimpleNamespace(
            success=True,
            extraction_result={"document_count": 2, "content_types": ["text", "image"]},
            split_result={
                "total_files_created": 4,
                "content_type_distribution": {"text": 2, "image": 2},
            },
            num_documents_indexed=4,
            num_texts=2,
            num_images=2,
            embedding_dimension=1024,
            vectorstore_path="vectorstore",
        ),
    )

    result = runner.invoke(
        cli.app,
        [
            "build-dataset",
            "pdf_vectorstore",
            str(config),
            "--source",
            str(source),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert captured["config"] == config
    assert captured["source_override"] == source
    assert captured["output_dir_override"] == output


def test_prepare_dataset_command_success(monkeypatch, tmp_path: Path) -> None:
    _patch_cli_common(monkeypatch)
    config = _config(tmp_path)
    vector_store = tmp_path / "vectorstore"
    dataset = tmp_path / "dataset"
    captured = _capture_init(
        monkeypatch,
        "BuildDatasetPrepareDatasetInput",
        "build_dataset_prepare_dataset",
        SimpleNamespace(
            success=True,
            dataset_entries=[{"id": "a"}, {"id": "b"}],
            failed_models=["M-2"],
            dataset_jsonl_path="dataset.jsonl",
        ),
    )

    result = runner.invoke(
        cli.app,
        [
            "build-dataset",
            "prepare-dataset",
            str(config),
            "--vector-store",
            str(vector_store),
            "--dataset",
            str(dataset),
        ],
    )

    assert result.exit_code == 0
    assert captured["config"] == config
    assert captured["vector_store_override"] == vector_store
    assert captured["dataset_override"] == dataset


def test_usd_command_batch_mode_success(monkeypatch, tmp_path: Path) -> None:
    _patch_cli_common(monkeypatch)
    usd_dir = tmp_path / "usd_input"
    usd_dir.mkdir()
    config = _config(
        tmp_path,
        "usd_dir: usd_input\noutput_dir: batch_output\n",
    )
    captured = _capture_init(
        monkeypatch,
        "BuildDatasetUsdInput",
        "build_dataset_usd",
        SimpleNamespace(
            success=True,
            batch_results={
                "scene_a.usd": {
                    "status": "success",
                    "num_prims": 3,
                    "num_images": 9,
                    "output_dir": str(tmp_path / "batch_output" / "scene_a"),
                },
                "scene_b.usd": {
                    "status": "success",
                    "num_prims": 2,
                    "num_images": 6,
                    "output_dir": str(tmp_path / "batch_output" / "scene_b"),
                },
            },
        ),
    )

    result = runner.invoke(cli.app, ["build-dataset", "usd", str(config)])

    assert result.exit_code == 0
    assert captured["config"] == config
    assert captured["source_override"] == usd_dir.resolve()
    assert captured["output_dir_override"] == (tmp_path / "batch_output").resolve()


def test_usd_command_single_file_success(monkeypatch, tmp_path: Path) -> None:
    _patch_cli_common(monkeypatch)
    config = _config(tmp_path, "usd_path: scene.usd\n")
    output_dir = tmp_path / "dataset_out"
    captured = _capture_init(
        monkeypatch,
        "BuildDatasetUsdInput",
        "build_dataset_usd",
        SimpleNamespace(
            success=True,
            dataset_path="dataset.json",
            num_prims=4,
            num_images=12,
        ),
    )

    result = runner.invoke(
        cli.app,
        [
            "build-dataset",
            "usd",
            str(config),
            "--output",
            str(output_dir),
            "--extract-metadata",
        ],
    )

    assert result.exit_code == 0
    assert captured["config"] == config
    assert captured["source_override"] is None
    assert captured["output_dir_override"] == output_dir
    assert captured["extract_metadata"] is True


def test_predict_and_apply_commands_delegate_to_pipeline(
    monkeypatch, tmp_path: Path
) -> None:
    _patch_cli_common(monkeypatch)
    config = _config(tmp_path)
    calls: list[dict[str, object]] = []

    def fake_pipeline(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(cli, "pipeline", fake_pipeline)

    predict_result = runner.invoke(cli.app, ["predict", str(config)])
    apply_result = runner.invoke(cli.app, ["apply", str(config)])

    assert predict_result.exit_code == 0
    assert apply_result.exit_code == 0
    assert calls[0]["only"] == "predict"
    assert calls[1]["only"] == "apply"
    assert all(call["config"] == config for call in calls)


def test_refine_material_command_reports_package_and_evidence(
    monkeypatch, tmp_path: Path
) -> None:
    _patch_cli_common(monkeypatch)
    config = _config(tmp_path)
    output_dir = tmp_path / "refined"
    captured = _capture_init(
        monkeypatch,
        "MaterialRefinementInput",
        "run_material_refinement_api",
        SimpleNamespace(
            success=True,
            approved=True,
            trial_count=4,
            best_material_dir=output_dir / "best_material",
            summary_path=output_dir / "summary.json",
        ),
    )

    result = runner.invoke(
        cli.app,
        ["refine-material", str(config), "--output-dir", str(output_dir)],
    )

    assert result.exit_code == 0
    assert captured == {"config": config, "output_dir_override": output_dir}


def test_refine_material_command_propagates_api_failure(
    monkeypatch, tmp_path: Path
) -> None:
    _patch_cli_common(monkeypatch)
    config = _config(tmp_path)
    _capture_init(
        monkeypatch,
        "MaterialRefinementInput",
        "run_material_refinement_api",
        SimpleNamespace(success=False, error=None),
    )

    result = runner.invoke(cli.app, ["refine-material", str(config)])

    assert result.exit_code == 1


def test_optimize_variations_command_reports_library_and_evidence(
    monkeypatch, tmp_path: Path
) -> None:
    _patch_cli_common(monkeypatch)
    config = _config(tmp_path)
    output_dir = tmp_path / "variations"
    captured = _capture_init(
        monkeypatch,
        "MaterialRefinementInput",
        "run_material_variations_api",
        SimpleNamespace(
            success=True,
            selected_count=3,
            material_library_path=output_dir / "material_library.usda",
            manifest_path=output_dir / "manifest.json",
        ),
    )

    result = runner.invoke(
        cli.app,
        ["optimize-variations", str(config), "--output-dir", str(output_dir)],
    )

    assert result.exit_code == 0
    assert captured == {"config": config, "output_dir_override": output_dir}


def test_optimize_variations_command_propagates_api_failure(
    monkeypatch, tmp_path: Path
) -> None:
    _patch_cli_common(monkeypatch)
    config = _config(tmp_path)
    _capture_init(
        monkeypatch,
        "MaterialRefinementInput",
        "run_material_variations_api",
        SimpleNamespace(success=False, error=None),
    )

    result = runner.invoke(cli.app, ["optimize-variations", str(config)])

    assert result.exit_code == 1
