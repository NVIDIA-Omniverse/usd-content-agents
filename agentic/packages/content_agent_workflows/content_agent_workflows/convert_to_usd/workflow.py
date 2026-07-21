# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agentic source-to-USD conversion workflow implementation."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from importlib import import_module
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

CONVERT_TO_USD_REPORT_SCHEMA_VERSION = (
    "content-agent-workflows.convert-to-usd-report.v1"
)
CONVERT_TO_USD_PROBE_SCHEMA_VERSION = "content-agent-workflows.convert-to-usd-probe.v1"
CONVERT_TO_USD_PREFLIGHT_SCHEMA_VERSION = (
    "content-agent-workflows.convert-to-usd-preflight.v1"
)
CONVERT_TO_USD_VALIDATION_SCHEMA_VERSION = (
    "content-agent-workflows.convert-to-usd-validation.v1"
)
CONVERT_TO_USD_MANIFEST_SCHEMA_VERSION = (
    "content-agent-workflows.convert-to-usd-manifest.v1"
)

CONVERT_TO_USD_SKILL = "convert-to-usd"
NEXT_STEP = "validate-usd-minimum"
DEFAULT_REFERENCE_ORDER = (
    "urdf-usd-converter",
    "mujoco-usd-converter",
    "usd-convert-cad",
)
USD_EXTENSIONS = frozenset({".usd", ".usda", ".usdc", ".usdz"})
OUTPUT_USD_FORMATS = ("usd", "usda", "usdc", "usdz")
OUTPUT_USD_FORMAT_SUFFIXES = {
    "usd": ".usd",
    "usda": ".usda",
    "usdc": ".usdc",
    "usdz": ".usdz",
}
USD_CONVERT_CAD_EXTENSIONS = frozenset(
    {
        ".3ds",
        ".3dxml",
        ".3mf",
        ".asm",
        ".catpart",
        ".catproduct",
        ".cgr",
        ".dae",
        ".dgn",
        ".dwg",
        ".dxf",
        ".e57",
        ".fbx",
        ".glb",
        ".gltf",
        ".iam",
        ".ifc",
        ".ifczip",
        ".iges",
        ".igs",
        ".ipt",
        ".jt",
        ".lxo",
        ".md5",
        ".obj",
        ".par",
        ".ply",
        ".prt",
        ".psm",
        ".pts",
        ".pwd",
        ".rfa",
        ".rvt",
        ".sab",
        ".sat",
        ".sldasm",
        ".sldprt",
        ".step",
        ".stl",
        ".stp",
        ".x_b",
        ".x_t",
        ".xmt",
        ".xmt_txt",
    }
)
CONVERTER_TOOLS = {
    "urdf-usd-converter": "urdf_usd_converter",
    "mujoco-usd-converter": "mujoco_usd_converter",
    "usd-convert-cad": "usd-convert-cad",
}
CONVERTER_PACKAGES = {
    "urdf-usd-converter": "urdf-usd-converter",
    "mujoco-usd-converter": "mujoco-usd-converter",
    "usd-convert-cad": "usd-convert-cad",
}
USD_CONVERT_CAD_REVISION = "4226fd49c06420adf193f821e2ddee805bb38eef"
USD_CONVERT_CAD_INSTALL_SPEC = (
    "git+https://github.com/NVIDIA-Omniverse/usd-convert-cad.git"
    f"@{USD_CONVERT_CAD_REVISION}"
)
CONVERTER_INSTALL_SPECS = {
    "urdf-usd-converter": ("urdf-usd-converter",),
    "mujoco-usd-converter": ("mujoco-usd-converter",),
    "usd-convert-cad": (
        USD_CONVERT_CAD_INSTALL_SPEC,
        "--extra-index-url",
        "https://pypi.nvidia.com",
    ),
}
CONVERTER_MODULES = {
    "urdf-usd-converter": "urdf_usd_converter",
    "mujoco-usd-converter": "mujoco_usd_converter",
    "usd-convert-cad": "usd_convert_cad",
}
CONVERTER_SOURCE_FORMATS = {
    "urdf-usd-converter": "urdf",
    "mujoco-usd-converter": "mjcf",
    "usd-convert-cad": "cad",
}
CONVERTER_INSTALL_TIMEOUT_S = 300.0
DIRECTORY_SOURCE_MAX_FILES = 4096
DIRECTORY_SOURCE_EXCLUDED_DIR_NAMES = frozenset(
    {
        ".git",
        ".hg",
        ".mypy_cache",
        ".nox",
        ".pytest_cache",
        ".ruff_cache",
        ".svn",
        ".tox",
        ".venv",
        "__pycache__",
        "node_modules",
        "venv",
    }
)

ConversionStatus = Literal["passed", "blocked", "failed"]
PreflightStatus = Literal["passed", "blocked"]
OutputUsdFormat = Literal["usd", "usda", "usdc", "usdz"]


class ConverterProbeResult(BaseModel):
    """Capability probe for one converter reference."""

    model_config = ConfigDict(extra="forbid")

    converter_skill: str = Field(min_length=1)
    converter_tool: str = Field(min_length=1)
    source_format: str = Field(min_length=1)
    supported: bool
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    install_hint: str = ""


class ConversionProbeArtifact(BaseModel):
    """Normalized probe artifact for the convert-to-USD router."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = CONVERT_TO_USD_PROBE_SCHEMA_VERSION
    source_asset_path: str
    reference_order: list[str]
    selected_converter: str | None = None
    probes: list[ConverterProbeResult] = Field(default_factory=list)


class ConverterPreflightReport(BaseModel):
    """Dependency preflight report for one source-to-USD request."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = CONVERT_TO_USD_PREFLIGHT_SCHEMA_VERSION
    status: PreflightStatus
    source_asset_path: str
    source_format: str = "unknown"
    converter_reference: str = ""
    converter_package: str = ""
    converter_tool: str = "none"
    dependency_available_before: bool = False
    dependency_available: bool = False
    install_requested: bool = False
    install_attempted: bool = False
    install_command: list[str] = Field(default_factory=list)
    install_hint: str = ""
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.status == "passed" and not self.errors


class ConversionReport(BaseModel):
    """Normalized source-to-USD conversion report."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = CONVERT_TO_USD_REPORT_SCHEMA_VERSION
    status: ConversionStatus
    source_asset_path: str
    source_format: str
    converter_skill: str
    converter_reference: str = ""
    converter_tool: str
    converter_command: list[str] = Field(default_factory=list)
    output_directory: str
    output_usd_path: str = ""
    output_format: str = "unknown"
    generated_files: list[str] = Field(default_factory=list)
    sidecar_inputs: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    diagnostics: list[dict[str, Any]] = Field(default_factory=list)
    install_hint: str = ""
    next_step: str = NEXT_STEP

    @property
    def passed(self) -> bool:
        return self.status == "passed" and not self.errors

    def to_markdown(self) -> str:
        command = " ".join(self.converter_command)
        lines = [
            "# Conversion Report",
            "",
            f"- Status: `{self.status}`",
            f"- Source asset: `{self.source_asset_path}`",
            f"- Source format: `{self.source_format}`",
            f"- Converter skill: `{self.converter_skill}`",
            f"- Converter reference: `{self.converter_reference or 'none'}`",
            f"- Converter tool: `{self.converter_tool}`",
            f"- Converter command: `{command}`",
            f"- Output directory: `{self.output_directory}`",
            f"- Output USD: `{self.output_usd_path}`",
            f"- Output format: `{self.output_format}`",
            f"- Next step: `{self.next_step}`",
            "",
            "## Generated Files",
            "",
        ]
        lines.extend(f"- `{path}`" for path in self.generated_files)
        if not self.generated_files:
            lines.append("- None")
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in self.warnings)
        if not self.warnings:
            lines.append("- None")
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in self.errors)
        if not self.errors:
            lines.append("- None")
        if self.install_hint:
            lines.extend(["", "## Install Hint", "", self.install_hint])
        lines.append("")
        return "\n".join(lines)


class ConvertToUsdWorkflowInput(BaseModel):
    """Input for the agentic convert-to-USD workflow."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    source_asset_path: Path
    output_dir: Path
    output_usd_path: Path | None = None
    output_format: OutputUsdFormat | None = None
    install_missing: bool = True
    reference_order: tuple[str, ...] = DEFAULT_REFERENCE_ORDER
    converter_timeout_s: float = 120.0
    fail_on_error: bool = False


class ConvertToUsdWorkflowResult(BaseModel):
    """Result and canonical artifacts from a convert-to-USD workflow."""

    model_config = ConfigDict(extra="forbid")

    success: bool
    source_asset: str
    output_dir: str
    output_usd_path: str | None = None
    selected_converter: str | None = None
    source_format: str = "unknown"
    request_path: str
    converter_probe_path: str
    conversion_report_path: str
    markdown_report_path: str
    validation_report_path: str
    manifest_path: str
    validation_status: str = "not_evaluated"
    error: str | None = None


def _as_json(model: BaseModel) -> dict[str, Any]:
    return cast(dict[str, Any], model.model_dump(mode="json"))


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def discover_generated_files(output_directory: Path | str) -> list[str]:
    """List generated files relative to an output directory."""

    output_dir = Path(output_directory)
    if not output_dir.exists():
        return []
    return sorted(
        str(path.relative_to(output_dir))
        for path in output_dir.rglob("*")
        if path.is_file()
    )


def discover_primary_usd(
    output_directory: Path | str,
    expected_output: Path | str,
) -> Path | None:
    """Find the primary USD output, preferring the expected converter path."""

    output_dir = Path(output_directory)
    expected = Path(expected_output)
    if expected.exists():
        return expected
    if not output_dir.exists():
        return None
    candidates = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in USD_EXTENSIONS
    )
    if len(candidates) == 1:
        return candidates[0]
    return None


def is_existing_usd(source_asset: Path | str) -> bool:
    """Return true when a source file is already an OpenUSD asset."""

    source_path = Path(source_asset)
    if not source_path.is_file():
        return False
    try:
        with source_path.open("rb") as file:
            header = file.read(16)
    except OSError:
        return False
    if header.startswith(b"#usda") or header.startswith(b"PXR-USDC"):
        return True
    if source_path.suffix.lower() == ".usdz" and zipfile.is_zipfile(source_path):
        try:
            with zipfile.ZipFile(source_path) as archive:
                for name in archive.namelist():
                    with archive.open(name) as member:
                        member_header = member.read(16)
                    if member_header.startswith(b"#usda") or member_header.startswith(
                        b"PXR-USDC"
                    ):
                        return True
        except (OSError, zipfile.BadZipFile):
            return False
    return False


def is_mujoco_source(source_asset: Path | str) -> bool:
    """Return true for MuJoCo XML/MJCF files."""

    try:
        for _event, element in ET.iterparse(source_asset, events=("start",)):
            return element.tag == "mujoco"
        return False
    except (ET.ParseError, OSError):
        return False


def normalize_output_format(
    output_format: str | None,
) -> OutputUsdFormat | None:
    """Normalize a user-facing output format token."""

    if output_format is None:
        return None
    normalized = output_format.lower().removeprefix(".")
    if normalized not in OUTPUT_USD_FORMATS:
        allowed = ", ".join(OUTPUT_USD_FORMATS)
        raise ValueError(
            f"unsupported output USD format: {output_format!r}. Expected one of: {allowed}"
        )
    return cast(OutputUsdFormat, normalized)


def output_format_for_path(path: Path | str) -> OutputUsdFormat | None:
    """Return the USD format implied by a path suffix."""

    suffix = Path(path).suffix.lower()
    for output_format, output_suffix in OUTPUT_USD_FORMAT_SUFFIXES.items():
        if suffix == output_suffix:
            return cast(OutputUsdFormat, output_format)
    return None


def default_output_usd_path(
    source_asset: Path | str,
    cwd: Path | None = None,
    *,
    output_format: str | None = None,
) -> Path:
    """Resolve the default output USD path for the file-oriented converter."""

    source_path = Path(source_asset)
    base_dir = (cwd or Path.cwd()).resolve()
    requested_format = normalize_output_format(output_format)
    if requested_format is None and is_existing_usd(source_path):
        return (base_dir / source_path.name).resolve()
    suffix = OUTPUT_USD_FORMAT_SUFFIXES[requested_format or "usda"]
    return (base_dir / f"{source_path.stem}{suffix}").resolve()


def resolve_output_usd_path(
    source_asset: Path | str,
    output_usd_path: Path | str | None = None,
    *,
    output_format: str | None = None,
    cwd: Path | None = None,
) -> Path:
    """Resolve and validate the requested output USD path."""

    requested_format = normalize_output_format(output_format)
    if output_usd_path is None:
        return default_output_usd_path(
            source_asset,
            cwd=cwd,
            output_format=requested_format,
        )

    resolved = Path(output_usd_path).expanduser().resolve()
    path_format = output_format_for_path(resolved)
    if path_format is None:
        allowed = ", ".join(sorted(USD_EXTENSIONS))
        raise ValueError(
            f"output USD path must end with a supported USD extension ({allowed}): {resolved}"
        )
    if requested_format is not None and path_format != requested_format:
        raise ValueError(
            "output USD path extension "
            f"`{resolved.suffix}` conflicts with requested output format "
            f"`{requested_format}`"
        )
    return resolved


def converter_reference_for_source_extension(
    source_asset: Path | str,
) -> str | None:
    """Return the Skill Hub converter reference implied by source extension."""

    source_path = Path(source_asset)
    suffix = source_path.suffix.lower()
    if suffix == ".urdf":
        return "urdf-usd-converter"
    if suffix == ".mjcf":
        return "mujoco-usd-converter"
    if suffix == ".xml" and is_mujoco_source(source_path):
        return "mujoco-usd-converter"
    if _cad_source_extension(source_path) is not None:
        return "usd-convert-cad"
    return None


def _cad_source_extension(source_asset: Path | str) -> str | None:
    suffixes = [suffix.lower() for suffix in Path(source_asset).suffixes]
    if not suffixes:
        return None
    if suffixes[-1] in USD_CONVERT_CAD_EXTENSIONS:
        return suffixes[-1]
    if suffixes[-1].removeprefix(".").isdigit():
        for suffix in reversed(suffixes[:-1]):
            if suffix in USD_CONVERT_CAD_EXTENSIONS:
                return suffix
    return None


def converter_package_for_source(source_asset: Path | str) -> str | None:
    """Return the optional converter package needed for a source asset."""

    converter_reference = converter_reference_for_source_extension(source_asset)
    if converter_reference is None:
        return None
    return CONVERTER_PACKAGES[converter_reference]


def _converter_tool_path(tool_name: str) -> str | None:
    tool_path = shutil.which(tool_name)
    if tool_path:
        return tool_path
    scripts_dir = Path(sys.executable).parent
    candidates = [scripts_dir / tool_name]
    if sys.platform == "win32":
        candidates.append(scripts_dir / f"{tool_name}.exe")
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def _dependency_available(converter_reference: str) -> bool:
    module_name = CONVERTER_MODULES.get(converter_reference)
    tool_name = CONVERTER_TOOLS.get(converter_reference)
    if converter_reference == "usd-convert-cad" and _usd_convert_cad_script():
        return True
    if tool_name:
        return bool(_converter_tool_path(tool_name))
    if module_name:
        try:
            import_module(module_name)
            return True
        except ImportError:
            return False
    return False


def _installer_command(converter_reference: str) -> list[str]:
    install_spec = CONVERTER_INSTALL_SPECS.get(converter_reference)
    if install_spec is None:
        install_spec = (CONVERTER_PACKAGES[converter_reference],)
    if shutil.which("uv") is not None:
        return ["uv", "pip", "install", "--python", sys.executable, *install_spec]
    return [sys.executable, "-m", "pip", "install", *install_spec]


def install_converter_package_for_source(
    source_asset: Path | str,
    *,
    timeout_s: float = CONVERTER_INSTALL_TIMEOUT_S,
) -> list[str] | None:
    """Install the converter package implied by the source extension if needed."""

    converter_reference = converter_reference_for_source_extension(source_asset)
    if converter_reference is None:
        return None
    if _dependency_available(converter_reference):
        return None
    command = _installer_command(converter_reference)
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "Timed out installing converter dependency with command: "
            + " ".join(command)
        ) from exc
    if completed.stdout:
        sys.stderr.write(completed.stdout)
    if completed.stderr:
        sys.stderr.write(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            "Failed to install converter dependency with command: " + " ".join(command)
        )
    return command


def preflight_convert_to_usd_dependencies(
    source_asset: Path | str,
    *,
    install_missing: bool = True,
    reference_order: tuple[str, ...] = DEFAULT_REFERENCE_ORDER,
) -> ConverterPreflightReport:
    """Prepare the converter dependency implied by a source asset."""

    source_path = Path(source_asset).resolve()
    warnings: list[str] = []

    if not source_path.exists():
        return ConverterPreflightReport(
            status="blocked",
            source_asset_path=str(source_path),
            errors=["source asset does not exist"],
        )

    if source_path.is_dir():
        selected_path, selected_probe, _probes, selection_warnings, report = (
            _directory_source_selection(
                source_path,
                Path.cwd().resolve(),
                reference_order=reference_order,
            )
        )
        warnings.extend(selection_warnings)
        if report is not None:
            return ConverterPreflightReport(
                status="blocked",
                source_asset_path=str(source_path),
                warnings=[*warnings, *report.warnings],
                errors=report.errors,
            )
        if selected_path is None:
            return ConverterPreflightReport(
                status="blocked",
                source_asset_path=str(source_path),
                warnings=warnings,
                errors=["directory source selection failed without a report"],
            )
        source_path = selected_path
    else:
        selected_probe = None

    if is_existing_usd(source_path):
        return ConverterPreflightReport(
            status="passed",
            source_asset_path=str(source_path),
            source_format="usd",
            converter_reference="existing-usd-passthrough",
            converter_package="",
            converter_tool="none",
            dependency_available_before=True,
            dependency_available=True,
            install_requested=False,
            warnings=warnings,
        )

    if selected_probe is None:
        selected_probe, _probes = select_converter(
            source_path,
            reference_order=reference_order,
        )
    if selected_probe is None:
        return ConverterPreflightReport(
            status="blocked",
            source_asset_path=str(source_path),
            warnings=warnings,
            errors=[
                "no enabled converter reference reported support for this source asset"
            ],
        )

    converter_reference = selected_probe.converter_skill
    dependency_available_before = _dependency_available(converter_reference)
    install_command: list[str] = []
    errors: list[str] = []

    if install_missing and not dependency_available_before:
        install_command = _installer_command(converter_reference)
        try:
            install_command = (
                install_converter_package_for_source(source_path) or install_command
            )
        except RuntimeError as exc:
            errors.append(str(exc))

    dependency_available = _dependency_available(converter_reference)
    if not dependency_available:
        errors.append(
            f"{selected_probe.converter_tool} CLI is required but was not found on PATH"
        )

    return ConverterPreflightReport(
        status="blocked" if errors else "passed",
        source_asset_path=str(source_path),
        source_format=selected_probe.source_format,
        converter_reference=converter_reference,
        converter_package=CONVERTER_PACKAGES[converter_reference],
        converter_tool=selected_probe.converter_tool,
        dependency_available_before=dependency_available_before,
        dependency_available=dependency_available,
        install_requested=install_missing,
        install_attempted=bool(install_command),
        install_command=install_command,
        install_hint=_install_hint(converter_reference),
        warnings=warnings,
        errors=errors,
    )


def _install_hint(converter_reference: str) -> str:
    if converter_reference not in CONVERTER_INSTALL_SPECS:
        return ""
    return "Install the converter dependency with: " + " ".join(
        _installer_command(converter_reference)
    )


def _usd_convert_cad_script() -> Path | None:
    root = os.getenv("USD_CONVERT_CAD_ROOT")
    if not root:
        return None
    script = Path(root).expanduser() / "convert.py"
    return script.resolve() if script.is_file() else None


def _converter_command(
    source_asset: Path,
    output_directory: Path,
    expected_output: Path,
    selected: ConverterProbeResult,
    *,
    single_file: bool,
) -> list[str]:
    if selected.converter_skill == "usd-convert-cad":
        script = _usd_convert_cad_script()
        if script is not None:
            return [
                sys.executable,
                str(script),
                str(source_asset),
                str(expected_output),
            ]
        tool_command = (
            _converter_tool_path(selected.converter_tool) or selected.converter_tool
        )
        return [
            tool_command,
            "--input",
            str(source_asset),
            "--output",
            str(expected_output),
        ]
    extra_args = ["--no-layer-structure"] if single_file else []
    tool_command = (
        _converter_tool_path(selected.converter_tool) or selected.converter_tool
    )
    return [
        tool_command,
        str(source_asset),
        str(output_directory),
        *extra_args,
    ]


def probe_converter(
    source_asset: Path | str,
    converter_skill: str,
) -> ConverterProbeResult:
    """Probe one converter reference for local source support."""

    source_path = Path(source_asset).resolve()
    tool = CONVERTER_TOOLS.get(converter_skill, "none")
    warnings: list[str] = []
    errors: list[str] = []

    if converter_skill == "urdf-usd-converter":
        supported = source_path.suffix.lower() == ".urdf"
        if not supported:
            warnings.append(
                "urdf_usd_converter expects a .urdf source, not "
                f"{source_path.suffix.lower() or 'unknown'}"
            )
        return ConverterProbeResult(
            converter_skill=converter_skill,
            converter_tool=tool,
            source_format="urdf" if supported else "unknown",
            supported=supported,
            warnings=warnings,
            install_hint=_install_hint(converter_skill),
        )

    if converter_skill == "mujoco-usd-converter":
        supported = is_mujoco_source(source_path)
        if not supported:
            warnings.append(
                "mujoco_usd_converter expects a source file with a <mujoco> XML root"
            )
        return ConverterProbeResult(
            converter_skill=converter_skill,
            converter_tool=tool,
            source_format="mjcf" if supported else "unknown",
            supported=supported,
            warnings=warnings,
            install_hint=_install_hint(converter_skill),
        )

    if converter_skill == "usd-convert-cad":
        supported = _cad_source_extension(source_path) is not None
        if not supported:
            warnings.append(
                "usd-convert-cad expects a CAD or mesh source extension, not "
                f"{source_path.suffix.lower() or 'unknown'}"
            )
        return ConverterProbeResult(
            converter_skill=converter_skill,
            converter_tool=tool,
            source_format="cad" if supported else "unknown",
            supported=supported,
            warnings=warnings,
            install_hint=_install_hint(converter_skill),
        )

    errors.append(
        f"converter reference is not enabled in this workflow: {converter_skill}"
    )
    return ConverterProbeResult(
        converter_skill=converter_skill,
        converter_tool=tool,
        source_format="unknown",
        supported=False,
        errors=errors,
    )


def select_converter(
    source_asset: Path | str,
    *,
    reference_order: tuple[str, ...] = DEFAULT_REFERENCE_ORDER,
) -> tuple[ConverterProbeResult | None, list[ConverterProbeResult]]:
    """Select the first supported converter reference by priority order."""

    probes = [
        probe_converter(source_asset, converter_skill)
        for converter_skill in reference_order
    ]
    selected = next((probe for probe in probes if probe.supported), None)
    return selected, probes


def _probe_warnings(probes: list[ConverterProbeResult]) -> list[str]:
    warnings: list[str] = []
    for probe in probes:
        status = "supported" if probe.supported else "not supported"
        warnings.append(
            f"Probe {probe.converter_skill}: {status} ({probe.source_format})"
        )
        warnings.extend(
            f"{probe.converter_skill}: {warning}" for warning in probe.warnings
        )
        warnings.extend(f"{probe.converter_skill}: {error}" for error in probe.errors)
    return warnings


def _report(
    *,
    status: ConversionStatus,
    source_asset: Path,
    source_format: str,
    converter_reference: str,
    converter_tool: str,
    converter_command: list[str],
    output_directory: Path,
    output_usd_path: Path | None = None,
    warnings: list[str] | None = None,
    errors: list[str] | None = None,
    install_hint: str = "",
) -> ConversionReport:
    return ConversionReport(
        status=status,
        source_asset_path=str(source_asset),
        source_format=source_format,
        converter_skill=CONVERT_TO_USD_SKILL,
        converter_reference=converter_reference,
        converter_tool=converter_tool,
        converter_command=converter_command,
        output_directory=str(output_directory),
        output_usd_path=str(output_usd_path) if output_usd_path else "",
        output_format=output_format_for_path(output_usd_path) or "unknown"
        if output_usd_path
        else "unknown",
        generated_files=discover_generated_files(output_directory),
        warnings=warnings or [],
        errors=errors or [],
        install_hint=install_hint,
    )


def _already_usd_report(
    source_asset: Path,
    output_directory: Path,
    *,
    output_usd_path: Path | None = None,
    warnings: list[str] | None = None,
) -> ConversionReport:
    return _report(
        status="passed",
        source_asset=source_asset,
        source_format="usd",
        converter_reference="existing-usd-passthrough",
        converter_tool="none",
        converter_command=[],
        output_directory=output_directory,
        output_usd_path=output_usd_path or source_asset,
        warnings=[
            *(warnings or []),
            "Source asset is already USD; conversion skipped.",
        ],
    )


def _unsupported_report(
    source_asset: Path,
    output_directory: Path,
    probes: list[ConverterProbeResult],
    *,
    warnings: list[str] | None = None,
) -> ConversionReport:
    return _report(
        status="blocked",
        source_asset=source_asset,
        source_format="unknown",
        converter_reference="",
        converter_tool="none",
        converter_command=[],
        output_directory=output_directory,
        warnings=[*(warnings or []), *_probe_warnings(probes)],
        errors=[
            "no enabled converter reference reported support for this source asset"
        ],
    )


def _directory_source_selection(
    source_directory: Path,
    output_directory: Path,
    *,
    reference_order: tuple[str, ...],
) -> tuple[
    Path | None,
    ConverterProbeResult | None,
    list[ConverterProbeResult],
    list[str],
    ConversionReport | None,
]:
    try:
        files = _directory_source_files(source_directory)
    except OSError as exc:
        report = _report(
            status="blocked",
            source_asset=source_directory,
            source_format="unknown",
            converter_reference="",
            converter_tool="none",
            converter_command=[],
            output_directory=output_directory,
            errors=[f"could not inspect directory source: {exc}"],
        )
        return None, None, [], [], report
    except RuntimeError as exc:
        report = _report(
            status="blocked",
            source_asset=source_directory,
            source_format="unknown",
            converter_reference="",
            converter_tool="none",
            converter_command=[],
            output_directory=output_directory,
            errors=[str(exc)],
        )
        return None, None, [], [], report

    selections: list[
        tuple[Path, ConverterProbeResult | None, list[ConverterProbeResult]]
    ] = []
    all_probes: list[ConverterProbeResult] = []
    for path in files:
        if is_existing_usd(path):
            selections.append((path, None, []))
            continue
        selected, probes = select_converter(path, reference_order=reference_order)
        all_probes.extend(probes)
        if selected is not None:
            selections.append((path, selected, probes))

    if not selections:
        detail = "no files" if not files else f"{len(files)} file(s)"
        report = _unsupported_report(
            source_directory,
            output_directory,
            all_probes,
            warnings=[
                f"Inspected directory source `{source_directory}` and found {detail}."
            ],
        )
        return None, None, all_probes, [], report

    if len(selections) > 1:
        candidates = ", ".join(
            path.relative_to(source_directory).as_posix() for path, _, _ in selections
        )
        report = _report(
            status="blocked",
            source_asset=source_directory,
            source_format="unknown",
            converter_reference="",
            converter_tool="none",
            converter_command=[],
            output_directory=output_directory,
            errors=[
                "directory source is ambiguous because multiple supported source "
                f"files were found: {candidates}. Pass one source file explicitly."
            ],
        )
        return None, None, all_probes, [], report

    selected_path, selected_probe, probes = selections[0]
    warning = (
        "Directory source contained exactly one supported source file; selected "
        f"`{selected_path.relative_to(source_directory).as_posix()}` for conversion."
    )
    return selected_path, selected_probe, probes, [warning], None


def _directory_source_files(source_directory: Path) -> list[Path]:
    def raise_walk_error(error: OSError) -> None:
        raise error

    files: list[Path] = []
    inspected_count = 0
    for root, dirnames, filenames in os.walk(
        source_directory,
        onerror=raise_walk_error,
    ):
        dirnames[:] = sorted(
            name for name in dirnames if name not in DIRECTORY_SOURCE_EXCLUDED_DIR_NAMES
        )
        root_path = Path(root)
        for filename in sorted(filenames):
            inspected_count += 1
            if inspected_count > DIRECTORY_SOURCE_MAX_FILES:
                raise RuntimeError(
                    "directory source inspection stopped after "
                    f"{DIRECTORY_SOURCE_MAX_FILES} files. Pass one source file "
                    "explicitly or reduce the directory contents."
                )
            files.append(root_path / filename)
    return files


def _run_converter_command(
    command: list[str],
    *,
    timeout_s: float,
) -> subprocess.CompletedProcess[str]:
    start_new_session = os.name != "nt"
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=start_new_session,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        try:
            if start_new_session:
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass
        stdout, stderr = process.communicate()
        raise subprocess.TimeoutExpired(
            exc.cmd,
            exc.timeout,
            output=stdout,
            stderr=stderr,
        ) from exc
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def _run_converter(
    source_asset: Path,
    output_directory: Path,
    selected: ConverterProbeResult,
    *,
    timeout_s: float,
    single_file: bool = False,
    warnings: list[str] | None = None,
) -> ConversionReport:
    tool = selected.converter_tool
    expected_output = output_directory / f"{source_asset.stem}.usda"
    command = _converter_command(
        source_asset,
        output_directory,
        expected_output,
        selected,
        single_file=single_file,
    )
    errors: list[str] = []
    all_warnings = list(warnings or [])

    if not source_asset.exists():
        errors.append(f"source asset does not exist: {source_asset}")
    elif selected.converter_skill == "urdf-usd-converter":
        if source_asset.suffix.lower() != ".urdf":
            errors.append(
                "unsupported URDF source format: "
                f"{source_asset.suffix.lower() or 'unknown'}"
            )
    elif selected.converter_skill == "mujoco-usd-converter":
        if not is_mujoco_source(source_asset):
            errors.append(
                "source asset is not a MuJoCo XML/MJCF file with a <mujoco> root"
            )
    elif selected.converter_skill == "usd-convert-cad":
        if _cad_source_extension(source_asset) is None:
            errors.append(
                "unsupported CAD conversion source format: "
                f"{source_asset.suffix.lower() or 'unknown'}"
            )
    else:
        errors.append(
            f"converter reference is not enabled in this workflow: {selected.converter_skill}"
        )

    if not _dependency_available(selected.converter_skill):
        errors.append(f"{tool} CLI is required but was not found on PATH")

    if errors:
        return _report(
            status="blocked",
            source_asset=source_asset,
            source_format=selected.source_format,
            converter_reference=selected.converter_skill,
            converter_tool=tool,
            converter_command=command,
            output_directory=output_directory,
            warnings=all_warnings,
            errors=errors,
            install_hint=_install_hint(selected.converter_skill),
        )

    output_directory.mkdir(parents=True, exist_ok=True)
    try:
        completed = _run_converter_command(
            command,
            timeout_s=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return _report(
            status="failed",
            source_asset=source_asset,
            source_format=selected.source_format,
            converter_reference=selected.converter_skill,
            converter_tool=tool,
            converter_command=command,
            output_directory=output_directory,
            warnings=all_warnings,
            errors=[
                f"converter timed out after {timeout_s}s: {selected.converter_skill}"
            ],
        )
    except OSError as exc:
        return _report(
            status="failed",
            source_asset=source_asset,
            source_format=selected.source_format,
            converter_reference=selected.converter_skill,
            converter_tool=tool,
            converter_command=command,
            output_directory=output_directory,
            warnings=all_warnings,
            errors=[f"failed to start converter {selected.converter_skill}: {exc}"],
        )

    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        errors.append(detail or f"{tool} exited with {completed.returncode}")

    primary_usd = discover_primary_usd(output_directory, expected_output)
    if primary_usd is None:
        errors.append(
            "converter did not produce an unambiguous primary USD output in: "
            f"{output_directory}"
        )
    elif primary_usd != expected_output:
        all_warnings.append(
            "Converter produced primary USD "
            f"`{primary_usd.name}` instead of expected `{expected_output.name}`."
        )

    return _report(
        status="failed" if errors else "passed",
        source_asset=source_asset,
        source_format=selected.source_format,
        converter_reference=selected.converter_skill,
        converter_tool=tool,
        converter_command=command,
        output_directory=output_directory,
        output_usd_path=primary_usd if primary_usd and primary_usd.exists() else None,
        warnings=all_warnings,
        errors=errors,
    )


def convert_to_usd(
    source_asset: Path | str,
    output_directory: Path | str,
    *,
    reference_order: tuple[str, ...] = DEFAULT_REFERENCE_ORDER,
    timeout_s: float = 120.0,
    single_file: bool = False,
) -> tuple[ConversionReport, ConversionProbeArtifact]:
    """Route one source asset to USD and return the report plus probe artifact."""

    source_path = Path(source_asset).resolve()
    output_dir = Path(output_directory).resolve()
    selected_probe: ConverterProbeResult | None = None
    probes: list[ConverterProbeResult] = []
    warnings: list[str] = []

    if not source_path.exists():
        report = _report(
            status="blocked",
            source_asset=source_path,
            source_format="unknown",
            converter_reference="",
            converter_tool="none",
            converter_command=[],
            output_directory=output_dir,
            errors=["source asset does not exist"],
        )
        return report, ConversionProbeArtifact(
            source_asset_path=str(source_path),
            reference_order=list(reference_order),
        )

    if source_path.is_dir():
        source_path, selected_probe, probes, warnings, report = (
            _directory_source_selection(
                source_path,
                output_dir,
                reference_order=reference_order,
            )
        )
        if report is not None:
            return report, ConversionProbeArtifact(
                source_asset_path=str(report.source_asset_path),
                reference_order=list(reference_order),
                selected_converter=None,
                probes=probes,
            )
        if source_path is None:
            raise RuntimeError("Directory source selection failed without a report.")

    if is_existing_usd(source_path):
        report = _already_usd_report(source_path, output_dir, warnings=warnings)
        return report, ConversionProbeArtifact(
            source_asset_path=str(source_path),
            reference_order=list(reference_order),
            selected_converter="existing-usd-passthrough",
            probes=probes,
        )

    if selected_probe is None:
        selected_probe, probes = select_converter(
            source_path,
            reference_order=reference_order,
        )
    if selected_probe is None:
        report = _unsupported_report(source_path, output_dir, probes, warnings=warnings)
        return report, ConversionProbeArtifact(
            source_asset_path=str(source_path),
            reference_order=list(reference_order),
            selected_converter=None,
            probes=probes,
        )

    selection_warnings = [
        *warnings,
        "Router selected "
        f"`{selected_probe.converter_skill}` from enabled converter capability probes.",
    ]
    supported = [probe.converter_skill for probe in probes if probe.supported]
    if len(supported) > 1:
        selection_warnings.append(
            "Multiple converter references reported support; selected by "
            "converter-reference priority: " + ", ".join(supported)
        )
    report = _run_converter(
        source_path,
        output_dir,
        selected_probe,
        timeout_s=timeout_s,
        single_file=single_file,
        warnings=selection_warnings,
    )
    return report, ConversionProbeArtifact(
        source_asset_path=str(source_path),
        reference_order=list(reference_order),
        selected_converter=selected_probe.converter_skill,
        probes=probes,
    )


def _copy_primary_usd(source_usd: Path, output_usd: Path) -> None:
    output_usd.parent.mkdir(parents=True, exist_ok=True)
    if source_usd.resolve() == output_usd.resolve():
        return
    shutil.copy2(source_usd, output_usd)


def _filesystem_usd_path(path: Path) -> Path:
    """Normalize USD asset identifiers back to filesystem paths."""

    path_text = str(path)
    if path_text.startswith("@") and path_text.endswith("@"):
        path_text = path_text[1:-1]
    elif path_text.endswith("@") and not path.exists():
        candidate = Path(path_text[:-1])
        if candidate.exists():
            return candidate
    return Path(path_text)


def _create_usdz_package(root_layer_path: Path, output_usd: Path) -> None:
    root_layer_path = _filesystem_usd_path(root_layer_path)
    output_usd.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{output_usd.stem}_",
        suffix=output_usd.suffix,
        dir=output_usd.parent,
        delete=False,
    ) as temp_file:
        temp_output = Path(temp_file.name)
    temp_output.unlink(missing_ok=True)
    try:
        from pxr import UsdUtils

        ok = UsdUtils.CreateNewUsdzPackage(str(root_layer_path), str(temp_output))
        if not ok or not temp_output.exists():
            raise RuntimeError(f"failed to create USDZ package: {output_usd}")
        if not zipfile.is_zipfile(temp_output):
            raise RuntimeError(
                f"CreateNewUsdzPackage wrote a non-ZIP file: {temp_output}"
            )
        temp_output.replace(output_usd)
    finally:
        temp_output.unlink(missing_ok=True)


def _export_usd_layer(source_usd: Path, output_usd: Path) -> None:
    source_usd = _filesystem_usd_path(source_usd)
    output_usd.parent.mkdir(parents=True, exist_ok=True)
    from pxr import Sdf

    source_layer = Sdf.Layer.FindOrOpen(str(source_usd))
    if source_layer is None:
        raise RuntimeError(f"failed to open USD layer for export: {source_usd}")
    if not source_layer.Export(str(output_usd)):
        raise RuntimeError(f"failed to export USD layer: {output_usd}")


def _write_primary_usd(source_usd: Path, output_usd: Path) -> None:
    """Write a primary USD output, converting layer/package format when needed."""

    source_usd = _filesystem_usd_path(source_usd)
    output_usd.parent.mkdir(parents=True, exist_ok=True)
    if source_usd.resolve() == output_usd.resolve():
        return

    output_format = output_format_for_path(output_usd)
    if output_format is None:
        allowed = ", ".join(OUTPUT_USD_FORMAT_SUFFIXES.values())
        raise ValueError(
            f"output USD path must end with a supported USD extension ({allowed}): "
            f"{output_usd}"
        )

    source_format = output_format_for_path(source_usd)
    if output_format == "usdz":
        if source_format == "usdz":
            shutil.copy2(source_usd, output_usd)
            return
        _create_usdz_package(source_usd, output_usd)
        return

    if source_format == output_format:
        shutil.copy2(source_usd, output_usd)
        return

    _export_usd_layer(source_usd, output_usd)


def _file_output_report(
    report: ConversionReport,
    *,
    output_usd_path: Path,
    generated_files: list[str] | None = None,
    sidecar_inputs: list[str] | None = None,
    extra_warnings: list[str] | None = None,
    errors: list[str] | None = None,
) -> ConversionReport:
    report_errors = [*report.errors, *(errors or [])]
    generated = generated_files if generated_files is not None else []
    output_written = (
        generated_files is not None
        and output_usd_path.name in generated_files
        and output_usd_path.exists()
    )
    status: ConversionStatus
    if report_errors:
        status = "failed" if report.status == "failed" else "blocked"
    elif output_written:
        status = "passed"
    else:
        status = "failed"
        report_errors.append(
            f"primary USD output was not generated by this run: {output_usd_path}"
        )
        output_written = False
    return report.model_copy(
        update={
            "status": status,
            "output_directory": str(output_usd_path.parent),
            "output_usd_path": str(output_usd_path) if output_written else "",
            "output_format": (output_format_for_path(output_usd_path) or "unknown")
            if output_written
            else "unknown",
            "generated_files": generated,
            "sidecar_inputs": sidecar_inputs
            if sidecar_inputs is not None
            else report.sidecar_inputs,
            "warnings": [*report.warnings, *(extra_warnings or [])],
            "errors": report_errors,
        }
    )


def _copy_generated_output_tree(
    workspace_dir: Path,
    primary_usd_path: Path,
    output_usd_path: Path,
) -> tuple[list[str], list[str]]:
    workspace_dir = workspace_dir.resolve()
    output_dir = output_usd_path.parent.resolve()
    primary_usd = _filesystem_usd_path(primary_usd_path).resolve()
    copied_sidecars: list[str] = []

    for source_path in sorted(
        (path for path in workspace_dir.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(workspace_dir).as_posix(),
    ):
        if source_path.resolve() == primary_usd:
            continue
        relative_path = source_path.relative_to(workspace_dir)
        target_path = output_dir / relative_path
        if target_path.resolve() == output_usd_path.resolve():
            continue
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
        copied_sidecars.append(relative_path.as_posix())

    _write_primary_usd(primary_usd, output_usd_path)
    return [output_usd_path.name, *copied_sidecars], copied_sidecars


def _workspace_sidecar_inputs(
    workspace_dir: Path,
    primary_usd_path: Path,
) -> list[str]:
    workspace_dir = workspace_dir.resolve()
    primary_usd = _filesystem_usd_path(primary_usd_path).resolve()
    return [
        path.relative_to(workspace_dir).as_posix()
        for path in sorted(
            (path for path in workspace_dir.rglob("*") if path.is_file()),
            key=lambda path: path.relative_to(workspace_dir).as_posix(),
        )
        if path.resolve() != primary_usd
    ]


def convert_source_to_usd_file(
    source_asset: Path | str,
    output_usd_path: Path | str | None = None,
    *,
    output_format: str | None = None,
    install_missing: bool = True,
    reference_order: tuple[str, ...] = DEFAULT_REFERENCE_ORDER,
    timeout_s: float = 120.0,
) -> tuple[ConversionReport, ConversionProbeArtifact]:
    """Convert one source asset to a requested USD file path.

    If ``output_usd_path`` is omitted, the output file is written to the current
    working directory using the source stem and a ``.usda`` suffix.
    """

    requested_source_path = Path(source_asset).resolve()

    if not requested_source_path.exists():
        output_usd = resolve_output_usd_path(
            requested_source_path,
            output_usd_path,
            output_format=output_format,
        )
        report = _report(
            status="blocked",
            source_asset=requested_source_path,
            source_format="unknown",
            converter_reference="",
            converter_tool="none",
            converter_command=[],
            output_directory=output_usd.parent,
            errors=["source asset does not exist"],
        )
        return report, ConversionProbeArtifact(
            source_asset_path=str(requested_source_path),
            reference_order=list(reference_order),
        )

    source_path = requested_source_path
    directory_warnings: list[str] = []
    directory_probes: list[ConverterProbeResult] = []
    if requested_source_path.is_dir():
        selected_path, _selected_probe, directory_probes, directory_warnings, report = (
            _directory_source_selection(
                requested_source_path,
                requested_source_path,
                reference_order=reference_order,
            )
        )
        if report is not None:
            output_usd = resolve_output_usd_path(
                requested_source_path,
                output_usd_path,
                output_format=output_format,
            )
            return _file_output_report(
                report,
                output_usd_path=output_usd,
                extra_warnings=directory_warnings,
            ), ConversionProbeArtifact(
                source_asset_path=str(report.source_asset_path),
                reference_order=list(reference_order),
                probes=directory_probes,
            )
        if selected_path is None:
            raise RuntimeError("Directory source selection failed without a report.")
        source_path = selected_path

    output_usd = resolve_output_usd_path(
        source_path,
        output_usd_path,
        output_format=output_format,
    )
    output_usd.parent.mkdir(parents=True, exist_ok=True)

    if is_existing_usd(source_path):
        try:
            _write_primary_usd(source_path, output_usd)
            warnings = ["Source asset is already USD; wrote requested output path."]
            errors: list[str] = []
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            warnings = []
            errors = [f"failed to write existing USD to requested output path: {exc}"]
        report = _already_usd_report(
            source_path,
            output_usd.parent,
            output_usd_path=output_usd if not errors else None,
            warnings=warnings,
        )
        report = _file_output_report(
            report,
            output_usd_path=output_usd,
            generated_files=[output_usd.name] if not errors else None,
            extra_warnings=directory_warnings,
            errors=errors,
        )
        return report, ConversionProbeArtifact(
            source_asset_path=str(source_path),
            reference_order=list(reference_order),
            selected_converter="existing-usd-passthrough",
            probes=directory_probes,
        )

    install_warnings: list[str] = []
    if install_missing:
        try:
            install_command = install_converter_package_for_source(source_path)
        except RuntimeError as exc:
            report = _report(
                status="blocked",
                source_asset=source_path,
                source_format="unknown",
                converter_reference=converter_reference_for_source_extension(
                    source_path
                )
                or "",
                converter_tool="none",
                converter_command=[],
                output_directory=output_usd.parent,
                errors=[str(exc)],
            )
            return report, ConversionProbeArtifact(
                source_asset_path=str(source_path),
                reference_order=list(reference_order),
            )
        if install_command is not None:
            install_warnings.append(
                "Installed converter dependency with: " + " ".join(install_command)
            )

    with tempfile.TemporaryDirectory(
        prefix=f"{source_path.stem}-convert-to-usd-",
        dir=output_usd.parent,
    ) as temp_dir:
        report, probe_artifact = convert_to_usd(
            source_path,
            Path(temp_dir),
            reference_order=reference_order,
            timeout_s=timeout_s,
            single_file=True,
        )
        report = report.model_copy(
            update={"warnings": [*directory_warnings, *report.warnings]}
        )
        if not report.passed or not report.output_usd_path:
            return _file_output_report(
                report,
                output_usd_path=output_usd,
                extra_warnings=install_warnings,
            ), probe_artifact

        try:
            if output_format_for_path(output_usd) == "usdz":
                sidecar_inputs = _workspace_sidecar_inputs(
                    Path(temp_dir),
                    Path(report.output_usd_path),
                )
                _write_primary_usd(Path(report.output_usd_path), output_usd)
                generated_files = [output_usd.name]
            else:
                generated_files, sidecar_inputs = _copy_generated_output_tree(
                    Path(temp_dir),
                    Path(report.output_usd_path),
                    output_usd,
                )
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            return _file_output_report(
                report,
                output_usd_path=output_usd,
                extra_warnings=install_warnings,
                errors=[
                    f"failed to write converted USD to requested output path: {exc}"
                ],
            ), probe_artifact

    return _file_output_report(
        report,
        output_usd_path=output_usd,
        generated_files=generated_files,
        sidecar_inputs=sidecar_inputs,
        extra_warnings=[
            *install_warnings,
            "Converter ran in a temporary workspace; generated sidecars and primary USD were written to the requested output directory.",
        ],
    ), probe_artifact


def _validation_report(report: ConversionReport) -> dict[str, Any]:
    output_usd_path = Path(report.output_usd_path) if report.output_usd_path else None
    output_exists = output_usd_path.exists() if output_usd_path is not None else False
    status = "pass" if report.passed and output_exists else "blocked"
    errors = [] if status == "pass" else ["No usable USD output was produced."]
    return {
        "schema_version": CONVERT_TO_USD_VALIDATION_SCHEMA_VERSION,
        "status": status,
        "output_usd_path": report.output_usd_path,
        "checks": [
            {
                "name": "output_usd_exists",
                "status": "pass" if output_exists else "fail",
                "summary": "The conversion report points to an existing USD output.",
            }
        ],
        "errors": [*report.errors, *errors],
        "warnings": report.warnings,
    }


def run_convert_to_usd_workflow(
    params: ConvertToUsdWorkflowInput,
) -> ConvertToUsdWorkflowResult:
    """Route a source asset to USD and write canonical workflow artifacts."""

    output_dir = params.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_usd_path = resolve_output_usd_path(
        params.source_asset_path,
        params.output_usd_path,
        output_format=params.output_format,
        cwd=output_dir,
    )

    request_path = _write_json(output_dir / "request.json", _as_json(params))
    report, probe_artifact = convert_source_to_usd_file(
        params.source_asset_path,
        output_usd_path,
        output_format=params.output_format,
        install_missing=params.install_missing,
        reference_order=params.reference_order,
        timeout_s=params.converter_timeout_s,
    )
    converter_probe_path = _write_json(
        output_dir / "converter_probe.json",
        _as_json(probe_artifact),
    )
    conversion_report_path = _write_json(
        output_dir / "conversion_report.json",
        _as_json(report),
    )
    markdown_report_path = output_dir / "conversion_report.md"
    markdown_report_path.write_text(report.to_markdown(), encoding="utf-8")
    validation_payload = _validation_report(report)
    validation_report_path = _write_json(
        output_dir / "validation_report.json",
        validation_payload,
    )
    manifest_path = _write_json(
        output_dir / "manifest.json",
        {
            "schema_version": CONVERT_TO_USD_MANIFEST_SCHEMA_VERSION,
            "source_asset_path": report.source_asset_path,
            "output_usd_path": report.output_usd_path,
            "output_format": report.output_format,
            "conversion_report": str(conversion_report_path),
            "converter_probe": str(converter_probe_path),
            "validation_report": str(validation_report_path),
            "next_step": report.next_step,
        },
    )

    if params.fail_on_error and not report.passed:
        error = "; ".join(report.errors) or "conversion did not pass"
    else:
        error = "; ".join(report.errors) if report.errors else None

    return ConvertToUsdWorkflowResult(
        success=report.passed,
        source_asset=report.source_asset_path,
        output_dir=str(output_dir),
        output_usd_path=report.output_usd_path or None,
        selected_converter=probe_artifact.selected_converter,
        source_format=report.source_format,
        request_path=str(request_path),
        converter_probe_path=str(converter_probe_path),
        conversion_report_path=str(conversion_report_path),
        markdown_report_path=str(markdown_report_path),
        validation_report_path=str(validation_report_path),
        manifest_path=str(manifest_path),
        validation_status=str(validation_payload["status"]),
        error=error,
    )
