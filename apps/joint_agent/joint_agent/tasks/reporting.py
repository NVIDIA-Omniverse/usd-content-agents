# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reporting tasks for Joint Agent."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from world_understanding.agentic.tasks import Task
from world_understanding.agentic.utils.html_report import (
    escape_html,
    format_images_html,
    format_system_prompt_section,
    validate_image_options,
)
from world_understanding.utils.object_store import ObjectStore

from joint_agent.functions.stage1_schema import normalize_stage1_prediction_payload

logger = logging.getLogger(__name__)


class GeneratePredictionReportTask(Task):
    """Generate an HTML report for predictions.

    Input context keys:
        - predictions_path: Path to predictions file
        - predictions_count: Number of predictions
        - failed_count: Number of failed predictions
        - token_stats: Token usage statistics
        - output_key: Key for classification output
        - dataset_path: Path to dataset file (for resolving relative image paths)
        - dataset: Original dataset entries (optional, loaded from dataset_path if not provided)
        - config: Optional configuration dict (for system prompt)
        - actual_system_prompt_used: Actual system prompt used (includes critique if enabled)
        - report_image_max_size: Optional max image size in pixels (e.g., 256 for 256x256)
        - report_image_format: Optional image format ('png' or 'jpeg', default: 'png')
        - report_image_quality: Optional JPEG quality (1-100, default: 85)

    Output context keys:
        - report_path: Path to generated HTML report
    """

    def __init__(
        self,
        image_max_size: int | None = None,
        image_format: str | None = None,
        image_quality: int | None = None,
    ):
        """Initialize the task.

        Args:
            image_max_size: Optional max image size in pixels (e.g., 256 for 256x256).
                           If set, images are resized to fit within this size.
                           Can be overridden by context key 'report_image_max_size'.
            image_format: Optional image format ('png' or 'jpeg', default: 'png').
                         Can be overridden by context key 'report_image_format'.
            image_quality: Optional JPEG quality (1-100, default: 85).
                          Only used if format is 'jpeg'.
                          Can be overridden by context key 'report_image_quality'.
        """
        self.name = "GeneratePredictionReport"
        self.description = "Generate HTML report for predictions"
        self.image_max_size = image_max_size
        self.image_format = image_format
        self.image_quality = image_quality

    def run(
        self, context: dict[str, Any], object_store: ObjectStore | None = None
    ) -> dict[str, Any]:
        """Generate the report.

        Args:
            context: Workflow context
            object_store: Optional object store

        Returns:
            Updated context with report path
        """
        predictions_path = context.get("predictions_path")
        if not predictions_path:
            logger.warning("No predictions_path in context, skipping report")
            return context

        predictions_path = Path(predictions_path)
        if not predictions_path.exists():
            logger.warning("Predictions file not found: %s", predictions_path)
            return context

        # Load predictions
        predictions = []
        with open(predictions_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    predictions.append(json.loads(line))

        # Get stats
        predictions_count = context.get("predictions_count", len(predictions))
        failed_count = context.get("failed_count", 0)
        token_stats = context.get("token_stats", {})
        output_key = context.get("output_key", "classification")

        # Load dataset for enriching predictions
        dataset_path = context.get("dataset_path")
        dataset = context.get("dataset", [])

        if object_store:
            dataset = object_store.get("dataset", dataset)

        # If dataset not in context/object_store, try loading from dataset_path
        if not dataset and dataset_path:
            dataset_path_obj = Path(dataset_path)
            if dataset_path_obj.exists():
                try:
                    with open(dataset_path_obj, encoding="utf-8") as f:
                        dataset = [json.loads(line) for line in f if line.strip()]
                    logger.debug("Loaded %d dataset entries for report", len(dataset))
                except Exception as e:
                    logger.warning("Failed to load dataset for report: %s", e)

        # Create dataset map by ID
        dataset_map = {entry.get("id"): entry for entry in dataset if entry.get("id")}

        # Get system prompt
        system_prompt = context.get("actual_system_prompt_used") or context.get(
            "config", {}
        ).get("system_prompt", "")

        # Get and validate image processing options
        image_max_size = (
            context.get("report_image_max_size")
            if context.get("report_image_max_size") is not None
            else self.image_max_size
        )
        image_format = context.get("report_image_format", self.image_format)
        image_quality = context.get("report_image_quality", self.image_quality)

        image_format, image_quality, image_max_size = validate_image_options(
            image_format, image_quality, image_max_size
        )

        # Generate HTML report
        report_path = predictions_path.parent / "report.html"

        html_content = self._generate_html(
            predictions=predictions,
            predictions_count=predictions_count,
            failed_count=failed_count,
            token_stats=token_stats,
            output_key=output_key,
            dataset_map=dataset_map,
            dataset_path=dataset_path,
            system_prompt=system_prompt,
            image_max_size=image_max_size,
            image_format=image_format,
            image_quality=image_quality,
        )

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info("Generated prediction report: %s", report_path)
        context["report_path"] = str(report_path)

        return context

    def _format_token_stats_html(self, token_stats: dict[str, Any]) -> str:
        """Format token usage statistics as HTML section.

        Args:
            token_stats: Token statistics dictionary from TokenTracker.get_stats()

        Returns:
            HTML string with token usage section
        """
        if not token_stats:
            return ""

        total_tokens = token_stats.get("total_tokens", 0)
        input_tokens = token_stats.get("total_input_tokens", 0)
        output_tokens = token_stats.get("total_output_tokens", 0)
        invocation_count = token_stats.get("invocation_count", 0)

        # If no meaningful stats, return empty
        if total_tokens == 0 and invocation_count == 0:
            return ""

        # Format by-model breakdown if available
        by_model_html = ""
        by_model = token_stats.get("by_model", {})
        if by_model:
            model_rows = []
            for model_name, stats in by_model.items():
                model_rows.append(
                    f"""
                    <tr>
                        <td>{escape_html(model_name)}</td>
                        <td>{stats.get("count", 0)}</td>
                        <td>{stats.get("input_tokens", 0):,}</td>
                        <td>{stats.get("output_tokens", 0):,}</td>
                        <td>{stats.get("total_tokens", 0):,}</td>
                    </tr>
                    """
                )
            by_model_html = f"""
            <h3>By Model</h3>
            <table style="margin-top: 1rem;">
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Calls</th>
                        <th>Input Tokens</th>
                        <th>Output Tokens</th>
                        <th>Total Tokens</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(model_rows)}
                </tbody>
            </table>
            """

        return f"""
        <h2>Token Usage</h2>
        <div class="stats">
            <div class="stat-card">
                <div class="stat-value">{total_tokens:,}</div>
                <div class="stat-label">Total Tokens</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{input_tokens:,}</div>
                <div class="stat-label">Input Tokens</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{output_tokens:,}</div>
                <div class="stat-label">Output Tokens</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{invocation_count}</div>
                <div class="stat-label">VLM Calls</div>
            </div>
        </div>
        {by_model_html}
        """

    def _generate_html(
        self,
        predictions: list[dict[str, Any]],
        predictions_count: int,
        failed_count: int,
        token_stats: dict[str, Any],
        output_key: str,
        dataset_map: dict[str, dict[str, Any]],
        dataset_path: str | None,
        system_prompt: str | None,
        image_max_size: int | None = None,
        image_format: str = "png",
        image_quality: int = 85,
    ) -> str:
        """Generate HTML content for the report.

        Args:
            predictions: List of predictions
            predictions_count: Number of successful predictions
            failed_count: Number of failed predictions
            token_stats: Token usage statistics
            output_key: Key for classification output
            dataset_map: Mapping from ID to dataset entry
            dataset_path: Path to dataset file (for resolving relative image paths)
            system_prompt: System prompt used for predictions
            image_max_size: Optional max image size in pixels
            image_format: Image format ('png' or 'jpeg')
            image_quality: JPEG quality (1-100)

        Returns:
            HTML content string
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Determine base directory for images
        base_dir = None
        if dataset_path:
            base_dir = Path(dataset_path).parent

        # Build predictions table rows (no limit - show all)
        rows = []
        for pred in predictions:
            pred_id = pred.get("id", "unknown")

            # Get dataset entry for this prediction
            dataset_entry = dataset_map.get(pred_id, {})

            # Extract input prompt from dataset entry
            # Support both v0.1 (text) and v0.2 (user_prompt) schemas
            input_prompt = dataset_entry.get("user_prompt") or dataset_entry.get(
                "text", ""
            )
            input_prompt_html = escape_html(input_prompt) if input_prompt else "N/A"

            # Handle images with metadata
            # Support both v0.1 and v0.2 schemas
            media_section = dataset_entry.get("media", {})
            if media_section and "images" in media_section:
                # v0.2 schema: extract paths and metadata from media.images
                media_images = media_section.get("images", [])
                image_paths = [
                    img.get("path") for img in media_images if img.get("path")
                ]
                image_metadata = [img.get("metadata", {}) for img in media_images]
            else:
                # v0.1 schema: separate images and image_metadata lists
                image_paths = dataset_entry.get("images", [])
                image_metadata = dataset_entry.get("image_metadata", [])

            # Format images HTML
            images_html = format_images_html(
                image_paths,
                base_dir,
                image_metadata,
                image_max_size=image_max_size,
                image_format=image_format,
                image_quality=image_quality,
            )

            classification = normalize_stage1_prediction_payload(
                pred.get(output_key, {}),
                output_key=output_key,
            )
            if isinstance(classification, dict):
                component_type = classification.get("component_type", "-")
                component_name = classification.get("component_name", "-")
                role = classification.get("role", "-")
                candidate = classification.get("is_articulation_candidate", "-")
                joint_hint = classification.get("joint_type_hint", "-")
                axis_hint = classification.get("axis_hint", "-")
                parent_hint = classification.get("parent_hint", "-")
                child_hint = classification.get("child_hint", "-")
                material = classification.get("material", "-")
                link_hints_html = (
                    "<div class='link-hints'>"
                    f"<span title='Parent hint'>P: {escape_html(str(parent_hint))}</span>"
                    f"<span title='Child hint'>C: {escape_html(str(child_hint))}</span>"
                    "</div>"
                )
                confidence = classification.get("confidence", "-")

                # Get original/full response
                original_response = classification.get("original_response", "")
            else:
                component_type = "-"
                component_name = str(classification)
                role = "-"
                candidate = "-"
                joint_hint = "-"
                axis_hint = "-"
                link_hints_html = "<span class='no-data'>-</span>"
                material = "-"
                confidence = "-"
                original_response = str(classification) if classification else ""

            original_response_html = (
                escape_html(original_response) if original_response else "N/A"
            )

            # Confidence badge color
            conf_class = {
                "high": "confidence-high",
                "medium": "confidence-medium",
                "low": "confidence-low",
            }.get(confidence.lower() if isinstance(confidence, str) else "", "")
            candidate_class = {
                True: "candidate-yes",
                False: "candidate-no",
            }.get(candidate, "candidate-unknown")
            candidate_label = (
                "yes"
                if candidate is True
                else "no"
                if candidate is False
                else str(candidate)
            )

            rows.append(
                f"""
                <tr>
                    <td title="{escape_html(pred_id)}">{escape_html(pred_id)}</td>
                    <td><div class="prompt-cell">{input_prompt_html}</div></td>
                    <td>{images_html}</td>
                    <td><span class="badge">{escape_html(str(component_type))}</span></td>
                    <td>{escape_html(str(component_name))}</td>
                    <td>{escape_html(str(role))}</td>
                    <td><span class="candidate {candidate_class}">{escape_html(candidate_label)}</span></td>
                    <td><span class="joint-hint">{escape_html(str(joint_hint))}</span></td>
                    <td>{escape_html(str(axis_hint))}</td>
                    <td>{link_hints_html}</td>
                    <td><span class="material">{escape_html(str(material))}</span></td>
                    <td><span class="confidence {conf_class}">{escape_html(str(confidence))}</span></td>
                    <td><div class="response-cell">{original_response_html}</div></td>
                </tr>
                """
            )

        rows_html = "\n".join(rows)

        # Calculate success rate
        total_count = predictions_count + failed_count
        success_rate = (predictions_count / total_count * 100) if total_count > 0 else 0

        # Token stats section
        token_section = self._format_token_stats_html(token_stats)

        # System prompt section
        system_prompt_section = format_system_prompt_section(system_prompt)

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Joint Agent - Prediction Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1600px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #007bff;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
            border-bottom: 1px solid #ddd;
            padding-bottom: 8px;
        }}
        h3 {{
            color: #666;
            margin-top: 20px;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border: 1px solid #e0e0e0;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #007bff;
        }}
        .stat-label {{
            color: #666;
            margin-top: 5px;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            font-size: 12px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
            vertical-align: top;
        }}
        th {{
            background: #007bff;
            color: white;
            font-weight: 600;
            position: sticky;
            top: 0;
            z-index: 10;
        }}
        th:first-child {{
            width: 180px;
            min-width: 150px;
            max-width: 250px;
        }}
        td:first-child {{
            width: 180px;
            min-width: 150px;
            max-width: 250px;
            font-size: 10px;
            word-wrap: break-word;
            word-break: break-all;
        }}
        tr:hover {{
            background: #f5f5f5;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            background: #e9ecef;
            font-size: 0.85em;
            font-weight: 500;
        }}
        .material {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            background: #d4edda;
            color: #155724;
            font-size: 0.85em;
        }}
        .joint-hint {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            background: #d1ecf1;
            color: #0c5460;
            font-size: 0.85em;
        }}
        .candidate {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.85em;
            font-weight: 500;
        }}
        .candidate-yes {{
            background: #d4edda;
            color: #155724;
        }}
        .candidate-no {{
            background: #e9ecef;
            color: #495057;
        }}
        .candidate-unknown {{
            background: #fff3cd;
            color: #856404;
        }}
        .link-hints {{
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            font-size: 0.85em;
        }}
        .link-hints span {{
            padding: 2px 6px;
            background: #f8f9fa;
            border-radius: 3px;
            border: 1px solid #dee2e6;
        }}
        .no-data {{
            color: #999;
        }}
        .confidence {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.85em;
            font-weight: 500;
        }}
        .confidence-high {{
            background: #d4edda;
            color: #155724;
        }}
        .confidence-medium {{
            background: #fff3cd;
            color: #856404;
        }}
        .confidence-low {{
            background: #f8d7da;
            color: #721c24;
        }}
        .prompt-cell {{
            max-width: 300px;
            max-height: 200px;
            overflow-y: auto;
            font-size: 12px;
            background: #f9f9f9;
            padding: 8px;
            border-radius: 4px;
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
        .response-cell {{
            max-width: 400px;
            max-height: 200px;
            overflow-y: auto;
            font-size: 12px;
            background: #f9f9f9;
            padding: 8px;
            border-radius: 4px;
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
        .image-container {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 5px;
        }}
        .image-with-caption {{
            display: flex;
            flex-direction: column;
            align-items: center;
            max-width: 150px;
        }}
        .image-caption {{
            font-size: 11px;
            color: #666;
            margin-top: 4px;
            text-align: center;
            line-height: 1.2;
            word-wrap: break-word;
        }}
        .image-thumbnail {{
            max-width: 100px;
            max-height: 100px;
            object-fit: cover;
            border: 1px solid #ddd;
            border-radius: 4px;
            cursor: pointer;
        }}
        .system-prompt-section {{
            margin: 30px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #007bff;
        }}
        .system-prompt-section h3 {{
            margin-top: 0;
            color: #333;
            font-size: 14px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .system-prompt-content {{
            margin-top: 15px;
            padding: 15px;
            background: white;
            border-radius: 4px;
            font-family: monospace;
            font-size: 10px;
            white-space: pre-wrap;
            word-wrap: break-word;
            max-height: 300px;
            overflow-y: auto;
            border: 1px solid #e0e0e0;
        }}
        .modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            overflow: auto;
            background-color: rgba(0,0,0,0.8);
        }}
        .modal-content {{
            margin: auto;
            display: block;
            max-width: 90%;
            max-height: 90%;
            margin-top: 50px;
        }}
        .close {{
            position: absolute;
            top: 15px;
            right: 35px;
            color: #f1f1f1;
            font-size: 40px;
            font-weight: bold;
            cursor: pointer;
        }}
        .footer {{
            margin-top: 30px;
            text-align: center;
            color: #666;
            font-size: 0.9em;
        }}
        .timestamp {{
            color: #666;
            font-size: 12px;
            margin-top: 10px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Joint Agent - Prediction Report</h1>
        <p class="timestamp">Generated: {now}</p>

        <h2>Summary Statistics</h2>
        <div class="stats">
            <div class="stat-card">
                <div class="stat-value">{predictions_count}</div>
                <div class="stat-label">Successful Predictions</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{failed_count}</div>
                <div class="stat-label">Failed Predictions</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{success_rate:.1f}%</div>
                <div class="stat-label">Success Rate</div>
            </div>
        </div>

        {token_section}

        {system_prompt_section}

        <h2>Detailed Predictions</h2>
        <table>
            <thead>
                <tr>
                    <th>ID</th>
                    <th>Input Prompt</th>
                    <th>Input Images</th>
                    <th>Type</th>
                    <th>Component Name</th>
                    <th>Role</th>
                    <th>Candidate</th>
                    <th>Joint Hint</th>
                    <th>Axis Hint</th>
                    <th>Parent/Child</th>
                    <th>Material</th>
                    <th>Confidence</th>
                    <th>Full Response</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>

        <div class="footer">
            <p>Joint Agent - Built on World Understanding Framework</p>
        </div>
    </div>

    <!-- Modal for image preview -->
    <div id="imageModal" class="modal">
        <span class="close" onclick="closeModal()">&times;</span>
        <img class="modal-content" id="modalImage">
    </div>

    <script>
        function showImage(src) {{
            var modal = document.getElementById('imageModal');
            var modalImg = document.getElementById('modalImage');
            modal.style.display = "block";
            modalImg.src = src;
        }}

        function closeModal() {{
            document.getElementById('imageModal').style.display = "none";
        }}

        // Close modal when clicking outside the image
        window.onclick = function(event) {{
            var modal = document.getElementById('imageModal');
            if (event.target == modal) {{
                modal.style.display = "none";
            }}
        }}
    </script>
</body>
</html>
        """

        return html
