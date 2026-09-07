# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime discovery for SimReady Foundation workflow adapters."""

from __future__ import annotations

import hashlib
import importlib.metadata as importlib_metadata
import json
import os
import platform
import re
import shlex
import shutil
import stat
import subprocess
import sys
import time
import tomllib
from pathlib import Path

from .models import (
    DEFAULT_SIMREADY_FOUNDATION_COMMIT,
    DEFAULT_SIMREADY_FOUNDATION_REF,
    DEFAULT_SIMREADY_FOUNDATION_REPO_URL,
    SimReadyRuntimeInfo,
)

SIMREADY_CACHE_DIR_ENV = "CONTENT_WORKFLOW_SIMREADY_CACHE_DIR"
SIMREADY_FOUNDATION_REF_ENV = "CONTENT_WORKFLOW_SIMREADY_FOUNDATION_REF"
SIMREADY_VENV_ENV = "CONTENT_WORKFLOW_SIMREADY_VENV"
SIMREADY_USD_PROVIDER_ENV = "CONTENT_WORKFLOW_SIMREADY_USD_PROVIDER"
SIMREADY_FOUNDATION_ROOT_ENV = "SIMREADY_FOUNDATION_ROOT"
SIMREADY_FOUNDATION_SPEC_ROOT_ENV = "SIMREADY_FOUNDATION_SPEC_ROOT"
SIMREADY_VENV_READY_MARKER = ".content-workflow-simready-installed"
SIMREADY_VENV_READY_SCHEMA = "content-agent-workflows.simready-venv.v3"
SIMREADY_USD_EXCHANGE_REQUIREMENT = "usd-exchange==2.3.0"
SIMREADY_USD_CORE_EXCLUDE = "usd-core"
SIMREADY_USD_CORE_OVERRIDE = "usd-core; python_version < '0'"
SIMREADY_VALIDATOR_CONSTRAINTS = (
    "Jinja2==3.1.6",
    "markdown-it-py==4.2.0",
    "MarkupSafe==3.0.3",
    "mdurl==0.1.2",
    "numpy==2.5.1",
    "omniverse-asset-validator==1.18.0",
    "omniverse-usd-profiles==1.10.22",
    "annotated-types==0.7.0",
    "pydantic==2.12.5",
    "pydantic-core==2.41.5",
    "simready-validate==2026.4.9",
    "typing-extensions==4.15.0",
    "typing-inspection==0.4.2",
)
SIMREADY_USD_CORE_REQUIREMENT = "usd-core==26.5"
SIMREADY_VALIDATOR_DISTRIBUTIONS = {
    "jinja2": "3.1.6",
    "markdown-it-py": "4.2.0",
    "markupsafe": "3.0.3",
    "mdurl": "0.1.2",
    "numpy": "2.5.1",
    "omniverse-asset-validator": "1.18.0",
    "omniverse-usd-profiles": "1.10.22",
    "annotated-types": "0.7.0",
    "pydantic": "2.12.5",
    "pydantic-core": "2.41.5",
    "simready-validate": "2026.4.9",
    "typing-extensions": "4.15.0",
    "typing-inspection": "0.4.2",
}
SIMREADY_SUBPROCESS_ENVIRONMENT_ALLOWLIST = frozenset(
    {
        "COMSPEC",
        "LANG",
        "LOGNAME",
        "PATHEXT",
        "REQUESTS_CA_BUNDLE",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "TMPDIR",
        "TZ",
        "USER",
        "WINDIR",
    }
)
SIMREADY_NETWORK_ENVIRONMENT_ALLOWLIST = frozenset(
    {
        "ALL_PROXY",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "NO_PROXY",
        "all_proxy",
        "http_proxy",
        "https_proxy",
        "no_proxy",
    }
)
_FULL_GIT_COMMIT = re.compile(r"[0-9a-fA-F]{40}")
_GIT_LFS_OID = re.compile(rb"oid sha256:([0-9a-f]{64})")
_GIT_LFS_SIZE = re.compile(rb"size ([0-9]+)")
_GIT_LFS_VERSION = b"version https://git-lfs.github.com/spec/v1"
_TRUSTED_GIT_EXECUTABLE_CANDIDATES = (
    Path("/usr/bin/git"),
    Path("/bin/git"),
    Path("/usr/local/bin/git"),
    Path("/opt/homebrew/bin/git"),
    Path(r"C:\Program Files\Git\cmd\git.exe"),
    Path(r"C:\Program Files\Git\bin\git.exe"),
)
_TRUSTED_GIT_LFS_EXECUTABLE_CANDIDATES = (
    Path("/usr/bin/git-lfs"),
    Path("/bin/git-lfs"),
    Path("/usr/local/bin/git-lfs"),
    Path("/opt/homebrew/bin/git-lfs"),
    Path(r"C:\Program Files\Git LFS\git-lfs.exe"),
    Path(r"C:\Program Files\Git\mingw64\bin\git-lfs.exe"),
)


def resolve_simready_runtime(
    *,
    foundation_root: Path | str | None = None,
    foundation_spec_root: Path | str | None = None,
    venv_path: Path | str | None = None,
    install_missing: bool = True,
    update_foundation: bool = False,
) -> SimReadyRuntimeInfo:
    """Resolve SimReady Foundation specs and validator executable."""

    warnings: list[str] = []
    errors: list[str] = []
    ref = os.getenv(SIMREADY_FOUNDATION_REF_ENV, DEFAULT_SIMREADY_FOUNDATION_REF)
    root, managed = _resolve_foundation_root(foundation_root, ref=ref)
    expected_commit = _expected_managed_commit(ref) if managed else None

    if managed:
        if invalid_ref := _invalid_git_ref(ref):
            errors.append(invalid_ref)
        elif expected_commit is None:
            errors.append(
                "Managed SimReady Foundation refs must be immutable: use the "
                f"pinned release {DEFAULT_SIMREADY_FOUNDATION_REF!r} or a full "
                f"40-character commit, not {ref!r}."
            )

    if (
        root is not None
        and managed
        and not errors
        and (install_missing or update_foundation)
    ):
        lock_path, lock_fd, lock_error = _acquire_foundation_lock(root)
        if lock_error:
            errors.append(lock_error)
        else:
            try:
                if (
                    root.exists()
                    and install_missing
                    and _foundation_commit(root) is None
                ):
                    if root.is_symlink():
                        errors.append(
                            "Refusing to replace a symlinked partial managed SimReady "
                            f"Foundation checkout: {root}"
                        )
                    else:
                        cleanup_error = _remove_managed_tree(root)
                        if cleanup_error:
                            errors.append(cleanup_error)
                if not root.exists() and install_missing and not errors:
                    clone_error = _clone_foundation(root, ref=ref)
                    if clone_error:
                        errors.append(clone_error)
                elif root.exists() and update_foundation:
                    errors.append(
                        "Managed SimReady Foundation checkouts are immutable; "
                        "select a new full commit ref instead of updating in place."
                    )
            finally:
                if lock_path is not None and lock_fd is not None:
                    _release_pid_lock(lock_path, lock_fd)

    if root is None:
        errors.append(
            "SimReady Foundation checkout is not configured. Set "
            f"{SIMREADY_FOUNDATION_ROOT_ENV} or pass --foundation-root."
        )
    elif not root.exists():
        errors.append(f"SimReady Foundation checkout does not exist: {root}")

    commit = _foundation_commit(root) if root is not None and root.exists() else None
    checkout_verified = not managed
    spec_tree_sha256: str | None = None
    if managed and commit != expected_commit:
        errors.append(
            "Managed SimReady Foundation release identity differs: "
            f"expected {expected_commit or 'an immutable commit'}, "
            f"observed {commit or 'unknown'}."
        )
    elif managed and root is not None and root.exists():
        checkout_errors, spec_tree_sha256 = _managed_checkout_verification(
            root,
            expected_commit=expected_commit,
        )
        errors.extend(checkout_errors)
        checkout_verified = not checkout_errors
    spec_root = _resolve_spec_root(root, foundation_spec_root)
    if managed and root is not None:
        expected_spec_root = (root / "nv_core" / "sr_specs" / "docs").resolve()
        if spec_root != expected_spec_root:
            errors.append(
                "Managed SimReady Foundation spec root must use the verified "
                f"checkout tree: expected {expected_spec_root}, observed {spec_root}."
            )
    specs_ready, spec_errors = _check_spec_root(spec_root)
    errors.extend(spec_errors)
    if spec_tree_sha256 is None and specs_ready and spec_root is not None:
        spec_tree_sha256 = _directory_tree_sha256(spec_root)
    requirements_sha256 = _foundation_requirements_sha256(root)

    selected_venv, managed_venv = _resolve_venv_path(
        venv_path,
        root,
        foundation_commit=commit,
    )
    marker_identity = _expected_venv_marker_identity(
        foundation_commit=commit,
        foundation_requirements_sha256=requirements_sha256,
    )
    require_verified_venv = managed or managed_venv
    validator = _validator_from_venv(
        selected_venv,
        require_ready_marker=require_verified_venv,
        expected_marker_identity=marker_identity,
    )
    install_command: list[str] = []

    if (
        validator is None
        and install_missing
        and root is not None
        and root.exists()
        and specs_ready
        and not errors
    ):
        install_command = _install_command(root, selected_venv)
        install_error = _prepare_validation_venv(
            install_command,
            expected_marker_identity=marker_identity,
        )
        if install_error:
            errors.append(install_error)
        validator = _validator_from_venv(
            selected_venv,
            require_ready_marker=True,
            expected_marker_identity=marker_identity,
        )

    if validator is None and not managed and not errors:
        path_validator = shutil.which("simready-validate")
        if path_validator:
            validator = Path(path_validator)
            warnings.append(
                "Using simready-validate from PATH because no dedicated "
                "SimReady validation venv executable was found."
            )

    runtime_ready = validator is not None and Path(validator).exists()
    if not runtime_ready:
        errors.append(
            "simready-validate executable is unavailable. Run SimReady preflight "
            "with install enabled or provide CONTENT_WORKFLOW_SIMREADY_VENV."
        )

    profiles = list_simready_profiles(spec_root) if specs_ready else []
    validator_identity = (
        _venv_runtime_identity(selected_venv) if runtime_ready else None
    )
    validator_runtime_verified = bool(
        validator_identity
        and (
            not require_verified_venv
            or _venv_ready_marker_valid(
                selected_venv,
                expected_marker_identity=marker_identity,
            )
        )
    )
    if managed and runtime_ready and not validator_runtime_verified:
        errors.append("Managed SimReady validator runtime identity is unverified.")

    return SimReadyRuntimeInfo(
        foundation_repo_url=DEFAULT_SIMREADY_FOUNDATION_REPO_URL,
        foundation_ref=ref,
        foundation_root=str(root) if root is not None else None,
        foundation_commit=commit,
        foundation_checkout_verified=checkout_verified,
        foundation_requirements_sha256=requirements_sha256,
        foundation_spec_tree_sha256=spec_tree_sha256,
        runtime_contract_sha256=_validator_contract_sha256(),
        foundation_spec_root=str(spec_root) if spec_root is not None else None,
        managed_foundation_checkout=managed,
        venv_path=str(selected_venv) if selected_venv is not None else None,
        validator_executable=str(validator) if validator is not None else None,
        validator_executable_sha256=(
            validator_identity.get("validator_executable_sha256")
            if validator_identity
            else None
        ),
        validator_distributions_sha256=(
            validator_identity.get("validator_distributions_sha256")
            if validator_identity
            else None
        ),
        validator_runtime_verified=validator_runtime_verified,
        install_command=install_command,
        available_profiles=profiles,
        specs_ready=specs_ready,
        runtime_ready=runtime_ready,
        warnings=_dedupe(warnings),
        errors=_dedupe(errors),
    )


def verify_simready_runtime_identity(runtime: SimReadyRuntimeInfo) -> list[str]:
    """Recheck a resolved managed runtime immediately around execution."""

    if not runtime.managed_foundation_checkout:
        return []
    errors: list[str] = []
    root = Path(runtime.foundation_root).resolve() if runtime.foundation_root else None
    expected_commit = _expected_managed_commit(runtime.foundation_ref)
    if root is None or not root.is_dir():
        return ["Managed SimReady Foundation checkout disappeared before execution."]
    if _foundation_commit(root) != expected_commit:
        errors.append("Managed SimReady Foundation commit changed before execution.")
    checkout_errors, spec_digest = _managed_checkout_verification(
        root,
        expected_commit=expected_commit,
    )
    errors.extend(checkout_errors)
    if spec_digest != runtime.foundation_spec_tree_sha256:
        errors.append("Managed SimReady Foundation spec-tree digest changed.")
    expected_spec_root = (root / "nv_core" / "sr_specs" / "docs").resolve()
    if (
        not runtime.foundation_spec_root
        or Path(runtime.foundation_spec_root).resolve() != expected_spec_root
    ):
        errors.append("Managed SimReady Foundation spec-root identity changed.")
    venv = Path(runtime.venv_path).resolve() if runtime.venv_path else None
    marker_identity = _expected_venv_marker_identity(
        foundation_commit=runtime.foundation_commit,
        foundation_requirements_sha256=runtime.foundation_requirements_sha256,
    )
    if venv is None or not _venv_ready_marker_valid(
        venv,
        expected_marker_identity=marker_identity,
    ):
        errors.append("Managed SimReady validator environment identity changed.")
    else:
        identity = _venv_runtime_identity(venv)
        if identity is None:
            errors.append("Managed SimReady validator environment became unreadable.")
        else:
            if (
                identity["validator_executable_sha256"]
                != runtime.validator_executable_sha256
            ):
                errors.append("Managed SimReady validator executable digest changed.")
            if (
                identity["validator_distributions_sha256"]
                != runtime.validator_distributions_sha256
            ):
                errors.append("Managed SimReady validator distribution digest changed.")
    return _dedupe(errors)


def build_validation_command(
    *,
    runtime: SimReadyRuntimeInfo,
    asset_path: Path,
    profile: str,
    profile_version: str,
    raw_report_path: Path,
) -> list[str]:
    """Build the Foundation `simready-validate` command."""

    if not runtime.validator_executable:
        raise RuntimeError("SimReady validator executable is not resolved.")
    if not runtime.foundation_spec_root:
        raise RuntimeError("SimReady Foundation spec root is not resolved.")
    spec_root = Path(runtime.foundation_spec_root)
    validator = Path(runtime.validator_executable)
    launcher = [str(validator)]
    if os.name == "nt":
        try:
            is_python_script = validator.read_bytes()[:2] == b"#!"
        except OSError:
            is_python_script = False
        if is_python_script:
            venv_python = (
                _venv_python(Path(runtime.venv_path)) if runtime.venv_path else None
            )
            launcher = [
                str(
                    venv_python
                    if venv_python and venv_python.is_file()
                    else sys.executable
                ),
                str(validator),
            ]
    return [
        *launcher,
        "--rules-path",
        str(spec_root / "capabilities"),
        "--features-path",
        str(spec_root / "features"),
        "--profiles-path",
        str(spec_root / "profiles" / "profiles.toml"),
        "--profile",
        profile,
        "--version",
        profile_version,
        "--output",
        str(raw_report_path),
        str(asset_path),
    ]


def list_simready_profiles(spec_root: Path | str | None) -> list[str]:
    """Return profile names from a Foundation `profiles.toml` file."""

    if spec_root is None:
        return []
    profiles_path = Path(spec_root) / "profiles" / "profiles.toml"
    if not profiles_path.exists():
        return []
    try:
        payload = tomllib.loads(profiles_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return []
    return sorted(str(key) for key in payload if isinstance(payload.get(key), dict))


def _resolve_foundation_root(
    foundation_root: Path | str | None, *, ref: str
) -> tuple[Path | None, bool]:
    if foundation_root is not None:
        return Path(foundation_root).expanduser().resolve(), False
    env_root = os.getenv(SIMREADY_FOUNDATION_ROOT_ENV)
    if env_root:
        return Path(env_root).expanduser().resolve(), False
    return _cache_dir() / "checkouts" / f"simready-foundation-{_safe_name(ref)}", True


def _resolve_spec_root(
    foundation_root: Path | None, foundation_spec_root: Path | str | None
) -> Path | None:
    if foundation_spec_root is not None:
        return Path(foundation_spec_root).expanduser().resolve()
    env_spec_root = os.getenv(SIMREADY_FOUNDATION_SPEC_ROOT_ENV)
    if env_spec_root:
        return Path(env_spec_root).expanduser().resolve()
    if foundation_root is not None:
        return foundation_root / "nv_core" / "sr_specs" / "docs"
    return None


def _resolve_venv_path(
    venv_path: Path | str | None,
    foundation_root: Path | None,
    *,
    foundation_commit: str | None = None,
) -> tuple[Path, bool]:
    if venv_path is not None:
        return Path(venv_path).expanduser().resolve(), False
    env_venv = os.getenv(SIMREADY_VENV_ENV)
    if env_venv:
        return Path(env_venv).expanduser().resolve(), False
    dependency_key = _foundation_dependency_identity(
        foundation_root,
        foundation_commit=foundation_commit,
    )
    provider_key = "usd-exchange" if _should_use_usd_exchange_provider() else "usd-core"
    platform_key = _runtime_platform_identity()
    digest = hashlib.sha256(
        (
            f"foundation={dependency_key}|provider={provider_key}|"
            f"platform={platform_key}|contract={_validator_contract_sha256()}"
        ).encode()
    ).hexdigest()[:12]
    return _cache_dir() / "venvs" / f"simready-foundation-{digest}", True


def _foundation_dependency_identity(
    foundation_root: Path | None,
    *,
    foundation_commit: str | None,
) -> str:
    commit_identity = f"git:{foundation_commit}" if foundation_commit else "git:unknown"
    requirements_identity = _foundation_requirements_sha256(foundation_root)
    return f"{commit_identity}|requirements:{requirements_identity or 'unresolved'}"


def _runtime_platform_identity() -> str:
    cache_tag = getattr(sys.implementation, "cache_tag", None) or "unknown"
    return "|".join((sys.platform, platform.machine().lower(), cache_tag))


def _validator_contract_sha256() -> str:
    contract = "\n".join(
        (
            SIMREADY_VENV_READY_SCHEMA,
            *SIMREADY_VALIDATOR_CONSTRAINTS,
            SIMREADY_USD_CORE_REQUIREMENT,
            SIMREADY_USD_EXCHANGE_REQUIREMENT,
        )
    )
    return hashlib.sha256(contract.encode()).hexdigest()


def _cache_dir() -> Path:
    configured = os.getenv(SIMREADY_CACHE_DIR_ENV)
    if configured:
        return Path(configured).expanduser().resolve()
    base = Path(os.getenv("XDG_CACHE_HOME", "~/.cache")).expanduser()
    return base / "content-agent-workflows" / "simready"


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-")
    return safe or "main"


def _expected_managed_commit(ref: str) -> str | None:
    if ref == DEFAULT_SIMREADY_FOUNDATION_REF:
        return DEFAULT_SIMREADY_FOUNDATION_COMMIT
    if _FULL_GIT_COMMIT.fullmatch(ref):
        return ref.lower()
    return None


def build_simready_subprocess_environment(
    *,
    executable_dir: Path | None = None,
    allow_network: bool = False,
    isolated_home: Path | None = None,
) -> dict[str, str]:
    """Build a minimal environment for managed SimReady subprocesses."""

    allowed = set(SIMREADY_SUBPROCESS_ENVIRONMENT_ALLOWLIST)
    if allow_network:
        allowed.update(SIMREADY_NETWORK_ENVIRONMENT_ALLOWLIST)
    environment = {
        name: value
        for name, value in os.environ.items()
        if name in allowed or name.startswith("LC_")
    }
    path_entries: list[str] = []
    if executable_dir is not None:
        path_entries.append(str(executable_dir))
    path_entries.extend(
        part
        for part in os.defpath.split(os.pathsep)
        if part and part != os.curdir and Path(part).is_absolute()
    )
    home = (
        isolated_home.expanduser().resolve()
        if isolated_home is not None
        else (_cache_dir() / "tool-home").resolve()
    )
    config_home = home / ".config"
    cache_home = home / ".cache"
    for path in (home, config_home, cache_home):
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    environment["PATH"] = os.pathsep.join(dict.fromkeys(path_entries))
    environment["HOME"] = str(home)
    environment["USERPROFILE"] = str(home)
    environment["XDG_CONFIG_HOME"] = str(config_home)
    environment["XDG_CACHE_HOME"] = str(cache_home)
    environment["APPDATA"] = str(config_home)
    environment["LOCALAPPDATA"] = str(cache_home)
    environment["PIP_CONFIG_FILE"] = os.devnull
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONHASHSEED"] = "0"
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONSAFEPATH"] = "1"
    environment["UV_NO_CONFIG"] = "1"
    return environment


def _managed_checkout_errors(
    root: Path,
    *,
    expected_commit: str | None = None,
) -> list[str]:
    errors, _ = _managed_checkout_verification(
        root,
        expected_commit=expected_commit or _foundation_commit(root),
    )
    return errors


def _managed_checkout_verification(
    root: Path,
    *,
    expected_commit: str | None,
) -> tuple[list[str], str | None]:
    if root.is_symlink():
        return [f"Managed SimReady Foundation checkout is a symlink: {root}"], None
    git_metadata = root / ".git"
    if not git_metadata.is_dir() or git_metadata.is_symlink():
        return [
            f"Managed SimReady Foundation checkout has unsafe Git metadata: {root}"
        ], None
    if expected_commit is None or not _FULL_GIT_COMMIT.fullmatch(expected_commit):
        return ["Managed SimReady Foundation expected commit is unavailable."], None
    git_executable = _managed_git_executable()
    if git_executable is None:
        return [
            "git is required to verify the managed SimReady Foundation checkout."
        ], None
    environment = _managed_git_environment(git_executable)
    try:
        status = subprocess.run(
            _managed_git_command(
                git_executable,
                "-C",
                str(root),
                "status",
                "--porcelain",
                "--untracked-files=all",
            ),
            check=False,
            capture_output=True,
            env=environment,
            text=True,
            timeout=30,
        )
        replacements = subprocess.run(
            _managed_git_command(
                git_executable,
                "-C",
                str(root),
                "replace",
                "--list",
            ),
            check=False,
            capture_output=True,
            env=environment,
            text=True,
            timeout=30,
        )
        index_flags = subprocess.run(
            _managed_git_command(
                git_executable,
                "-C",
                str(root),
                "ls-files",
                "-v",
                "--",
                "requirements.txt",
                "nv_core/validator_sample",
                "nv_core/sr_specs/docs",
            ),
            check=False,
            capture_output=True,
            env=environment,
            text=True,
            timeout=30,
        )
        tree = subprocess.run(
            _managed_git_command(
                git_executable,
                "-C",
                str(root),
                "ls-tree",
                "-r",
                "-z",
                expected_commit,
                "--",
                "requirements.txt",
                "nv_core/validator_sample",
                "nv_core/sr_specs/docs",
            ),
            check=False,
            capture_output=True,
            env=environment,
            timeout=30,
        )
        object_format = subprocess.run(
            _managed_git_command(
                git_executable,
                "-C",
                str(root),
                "rev-parse",
                "--show-object-format",
            ),
            check=False,
            capture_output=True,
            env=environment,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [f"Could not verify managed SimReady Foundation checkout: {exc}"], None
    errors: list[str] = []
    if status.returncode != 0:
        errors.append(
            "Could not inspect managed SimReady Foundation checkout status: "
            + status.stderr.strip()
        )
    elif status.stdout.strip():
        errors.append(
            "Managed SimReady Foundation checkout contains local changes: "
            + status.stdout.strip().replace("\n", "; ")
        )
    if replacements.returncode != 0:
        errors.append(
            "Could not inspect managed Foundation replacement refs: "
            + replacements.stderr.strip()
        )
    elif replacements.stdout.strip():
        errors.append("Managed SimReady Foundation checkout contains replacement refs.")
    if index_flags.returncode != 0:
        errors.append(
            "Could not inspect managed Foundation index flags: "
            + index_flags.stderr.strip()
        )
    else:
        special = [
            line for line in index_flags.stdout.splitlines() if line and line[0] != "H"
        ]
        if special:
            errors.append(
                "Managed SimReady Foundation source files use special index flags: "
                + "; ".join(special)
            )
    if tree.returncode != 0:
        errors.append(
            "Could not read managed Foundation commit tree: "
            + tree.stderr.decode("utf-8", errors="replace").strip()
        )
        return errors, None
    algorithm = (
        object_format.stdout.strip() if object_format.returncode == 0 else "sha1"
    )
    if algorithm not in hashlib.algorithms_available:
        errors.append(
            f"Unsupported Git object format for Foundation verification: {algorithm}"
        )
        return errors, None
    try:
        expected_blobs = _parse_git_tree_blobs(tree.stdout)
        actual_paths = _managed_source_paths(root)
    except (OSError, ValueError) as exc:
        errors.append(f"Could not enumerate managed Foundation source bytes: {exc}")
        return errors, None
    expected_paths = set(expected_blobs)
    actual_relative_paths = set(actual_paths)
    if actual_relative_paths != expected_paths:
        missing = sorted(expected_paths - actual_relative_paths)
        extra = sorted(actual_relative_paths - expected_paths)
        errors.append(
            "Managed SimReady Foundation source file set differs from the pinned commit: "
            f"missing={missing}, extra={extra}"
        )
    mismatched_payloads: dict[str, bytes] = {}
    for relative in sorted(expected_paths & actual_relative_paths):
        path = actual_paths[relative]
        if path.is_symlink():
            errors.append(f"Managed Foundation source path is a symlink: {relative}")
            continue
        try:
            payload = path.read_bytes()
        except OSError as exc:
            errors.append(f"Could not read managed Foundation source {relative}: {exc}")
            continue
        if _git_lfs_pointer_target(payload) is not None:
            errors.append(
                "Managed Foundation source is an unresolved Git LFS pointer: "
                + relative
            )
            continue
        if _git_blob_oid(payload, algorithm=algorithm) != expected_blobs[relative]:
            mismatched_payloads[relative] = payload

    if mismatched_payloads:
        mismatched_oids = [expected_blobs[path] for path in mismatched_payloads]
        committed_payloads, blob_error = _read_git_blobs(
            root,
            mismatched_oids,
            git_executable=git_executable,
            environment=environment,
        )
        if blob_error:
            errors.append(blob_error)
            committed_payloads = {}
        for relative, payload in mismatched_payloads.items():
            committed = committed_payloads.get(expected_blobs[relative])
            if committed is not None and _matches_git_lfs_pointer(committed, payload):
                continue
            errors.append(
                "Managed SimReady Foundation source bytes differ from the pinned commit: "
                + relative
            )
    spec_root = root / "nv_core" / "sr_specs" / "docs"
    if spec_root.is_symlink():
        errors.append("Managed SimReady Foundation spec root is a symlink.")
    spec_digest = _directory_tree_sha256(spec_root) if spec_root.is_dir() else None
    return errors, spec_digest


def _parse_git_tree_blobs(payload: bytes) -> dict[str, str]:
    blobs: dict[str, str] = {}
    for raw_entry in payload.split(b"\0"):
        if not raw_entry:
            continue
        header, separator, raw_path = raw_entry.partition(b"\t")
        fields = header.split()
        if separator != b"\t" or len(fields) != 3 or fields[1] != b"blob":
            raise ValueError("managed Foundation commit tree contains a non-blob entry")
        relative = os.fsdecode(raw_path)
        if relative in blobs:
            raise ValueError(f"duplicate managed Foundation commit path: {relative}")
        blobs[relative] = fields[2].decode("ascii")
    return blobs


def _managed_source_paths(root: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for relative in ("requirements.txt",):
        requirements = root / relative
        if requirements.exists() or requirements.is_symlink():
            paths[relative] = requirements
    for source_root in (
        root / "nv_core" / "validator_sample",
        root / "nv_core" / "sr_specs" / "docs",
    ):
        if not source_root.is_dir():
            continue
        for directory, directories, filenames in os.walk(
            source_root, followlinks=False
        ):
            current = Path(directory)
            for name in list(directories):
                candidate = current / name
                if candidate.is_symlink():
                    paths[candidate.relative_to(root).as_posix()] = candidate
                    directories.remove(name)
            for name in filenames:
                candidate = current / name
                relative = candidate.relative_to(root)
                if "__pycache__" in relative.parts or candidate.suffix in {
                    ".pyc",
                    ".pyo",
                }:
                    continue
                paths[relative.as_posix()] = candidate
    return paths


def _git_blob_oid(payload: bytes, *, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    digest.update(f"blob {len(payload)}\0".encode("ascii"))
    digest.update(payload)
    return digest.hexdigest()


def _matches_git_lfs_pointer(pointer: bytes, payload: bytes) -> bool:
    """Return whether payload is the exact object named by a Git LFS pointer."""

    target = _git_lfs_pointer_target(pointer)
    if target is None:
        return False
    expected_sha256, expected_size = target
    return (
        len(payload) == expected_size
        and hashlib.sha256(payload).hexdigest() == expected_sha256
    )


def _git_lfs_pointer_target(pointer: bytes) -> tuple[str, int] | None:
    if len(pointer) > 8192:
        return None
    lines = pointer.rstrip(b"\n").splitlines()
    if not lines or lines[0] != _GIT_LFS_VERSION:
        return None
    oid_matches = [match for line in lines if (match := _GIT_LFS_OID.fullmatch(line))]
    size_matches = [match for line in lines if (match := _GIT_LFS_SIZE.fullmatch(line))]
    if len(oid_matches) != 1 or len(size_matches) != 1:
        return None
    return (
        oid_matches[0].group(1).decode("ascii"),
        int(size_matches[0].group(1)),
    )


def _read_git_blobs(
    root: Path,
    object_ids: list[str],
    *,
    git_executable: Path,
    environment: dict[str, str],
) -> tuple[dict[str, bytes], str | None]:
    """Read pinned Git blobs in one bounded subprocess."""

    unique_ids = list(dict.fromkeys(object_ids))
    if not unique_ids:
        return {}, None
    try:
        completed = subprocess.run(
            _managed_git_command(
                git_executable,
                "-C",
                str(root),
                "cat-file",
                "--batch",
            ),
            input=("\n".join(unique_ids) + "\n").encode("ascii"),
            check=False,
            capture_output=True,
            env=environment,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {}, f"Could not read managed Foundation commit blobs: {exc}"
    if completed.returncode != 0:
        return {}, (
            "Could not read managed Foundation commit blobs: "
            + completed.stderr.decode("utf-8", errors="replace").strip()
        )
    try:
        return _parse_git_batch_blobs(completed.stdout, expected_ids=unique_ids), None
    except ValueError as exc:
        return {}, f"Could not parse managed Foundation commit blobs: {exc}"


def _parse_git_batch_blobs(
    payload: bytes, *, expected_ids: list[str]
) -> dict[str, bytes]:
    blobs: dict[str, bytes] = {}
    offset = 0
    for expected_id in expected_ids:
        header_end = payload.find(b"\n", offset)
        if header_end < 0:
            raise ValueError("truncated Git batch header")
        header = payload[offset:header_end].split()
        if len(header) != 3 or header[1] != b"blob":
            raise ValueError("unexpected Git batch object type")
        object_id = header[0].decode("ascii")
        if object_id != expected_id:
            raise ValueError("Git batch object order differs")
        try:
            size = int(header[2])
        except ValueError as exc:
            raise ValueError("invalid Git batch object size") from exc
        content_start = header_end + 1
        content_end = content_start + size
        if (
            content_end >= len(payload)
            or payload[content_end : content_end + 1] != b"\n"
        ):
            raise ValueError("truncated Git batch object")
        blobs[object_id] = payload[content_start:content_end]
        offset = content_end + 1
    if offset != len(payload):
        raise ValueError("unexpected trailing Git batch data")
    return blobs


def _directory_tree_sha256(root: Path) -> str | None:
    if not root.is_dir():
        return None
    rows: list[dict[str, object]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        try:
            payload = path.read_bytes()
        except OSError:
            return None
        rows.append(
            {
                "path": relative.as_posix(),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return hashlib.sha256(
        json.dumps(rows, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def _check_spec_root(spec_root: Path | None) -> tuple[bool, list[str]]:
    if spec_root is None:
        return False, ["SimReady Foundation spec root is not configured."]
    missing: list[str] = []
    for relative in (
        "capabilities",
        "features",
        "profiles/profiles.toml",
    ):
        if not (spec_root / relative).exists():
            missing.append(str(spec_root / relative))
    if missing:
        return False, [
            "SimReady Foundation spec files are missing: " + ", ".join(missing)
        ]
    return True, []


def _validator_from_venv(
    venv_path: Path | None,
    *,
    require_ready_marker: bool = False,
    expected_marker_identity: dict[str, object] | None = None,
) -> Path | None:
    if venv_path is None:
        return None
    if require_ready_marker and not _venv_ready_marker_valid(
        venv_path,
        expected_marker_identity=expected_marker_identity,
    ):
        return None
    executable = (
        "simready-validate.exe" if sys.platform == "win32" else "simready-validate"
    )
    scripts_dir = "Scripts" if sys.platform == "win32" else "bin"
    candidate = venv_path / scripts_dir / executable
    return candidate if candidate.exists() else None


def _venv_ready_marker(venv_path: Path) -> Path:
    return venv_path / SIMREADY_VENV_READY_MARKER


def _expected_venv_marker_identity(
    *,
    foundation_commit: str | None,
    foundation_requirements_sha256: str | None,
) -> dict[str, object]:
    use_usd_exchange = _should_use_usd_exchange_provider()
    return {
        "schema": SIMREADY_VENV_READY_SCHEMA,
        "foundation_commit": foundation_commit,
        "foundation_requirements_sha256": foundation_requirements_sha256,
        "platform_identity": _runtime_platform_identity(),
        "runtime_contract_sha256": _validator_contract_sha256(),
        "usd_provider": "usd-exchange" if use_usd_exchange else "usd-core",
        "usd_provider_requirement": (
            SIMREADY_USD_EXCHANGE_REQUIREMENT
            if use_usd_exchange
            else SIMREADY_USD_CORE_REQUIREMENT
        ),
    }


def _normalized_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _expected_distribution_inventory(*, usd_provider: str) -> dict[str, str]:
    inventory = dict(SIMREADY_VALIDATOR_DISTRIBUTIONS)
    if usd_provider == "usd-exchange":
        inventory["usd-exchange"] = SIMREADY_USD_EXCHANGE_REQUIREMENT.partition("==")[2]
    else:
        inventory["usd-core"] = SIMREADY_USD_CORE_REQUIREMENT.partition("==")[2]
    return dict(sorted(inventory.items()))


def _venv_site_packages(venv_path: Path) -> list[Path]:
    candidates = [venv_path / "Lib" / "site-packages"]
    candidates.extend(sorted((venv_path / "lib").glob("python*/site-packages")))
    return [path for path in candidates if path.is_dir()]


def _venv_runtime_identity(venv_path: Path) -> dict[str, str] | None:
    validator = _validator_from_venv(venv_path)
    site_packages = _venv_site_packages(venv_path)
    if validator is None or not validator.is_file() or len(site_packages) != 1:
        return None
    try:
        validator_sha256 = hashlib.sha256(validator.read_bytes()).hexdigest()
        distributions: dict[str, str] = {}
        for distribution in importlib_metadata.distributions(
            path=[str(site_packages[0])]
        ):
            raw_name = distribution.metadata.get("Name")
            if not isinstance(raw_name, str) or not raw_name:
                return None
            name = _normalized_distribution_name(raw_name)
            if name in distributions:
                return None
            distributions[name] = distribution.version
    except (OSError, UnicodeError):
        return None
    normalized = dict(sorted(distributions.items()))
    return {
        "validator_executable_sha256": validator_sha256,
        "validator_distributions_sha256": hashlib.sha256(
            json.dumps(normalized, separators=(",", ":"), sort_keys=True).encode(
                "utf-8"
            )
        ).hexdigest(),
        "validator_distributions_json": json.dumps(
            normalized,
            separators=(",", ":"),
            sort_keys=True,
        ),
    }


def _venv_ready_marker_valid(
    venv_path: Path,
    *,
    expected_marker_identity: dict[str, object] | None = None,
) -> bool:
    try:
        payload = json.loads(_venv_ready_marker(venv_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    expected = expected_marker_identity or {
        "schema": SIMREADY_VENV_READY_SCHEMA,
        "platform_identity": _runtime_platform_identity(),
        "runtime_contract_sha256": _validator_contract_sha256(),
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        return False
    usd_provider = payload.get("usd_provider")
    runtime_identity = _venv_runtime_identity(venv_path)
    if not isinstance(usd_provider, str) or runtime_identity is None:
        return False
    try:
        distributions = json.loads(runtime_identity["validator_distributions_json"])
    except json.JSONDecodeError:
        return False
    return (
        distributions == _expected_distribution_inventory(usd_provider=usd_provider)
        and payload.get("validator_executable_sha256")
        == runtime_identity["validator_executable_sha256"]
        and payload.get("validator_distributions_sha256")
        == runtime_identity["validator_distributions_sha256"]
    )


def _write_venv_ready_marker(
    venv_path: Path,
    *,
    expected_marker_identity: dict[str, object],
) -> None:
    runtime_identity = _venv_runtime_identity(venv_path)
    if runtime_identity is None:
        raise OSError("SimReady validator runtime identity could not be inspected")
    usd_provider = expected_marker_identity.get("usd_provider")
    distributions = json.loads(runtime_identity["validator_distributions_json"])
    if not isinstance(
        usd_provider, str
    ) or distributions != _expected_distribution_inventory(usd_provider=usd_provider):
        raise OSError("SimReady validator installed distribution set differs")
    payload = {
        **expected_marker_identity,
        "validator_executable_sha256": runtime_identity["validator_executable_sha256"],
        "validator_distributions_sha256": runtime_identity[
            "validator_distributions_sha256"
        ],
    }
    _venv_ready_marker(venv_path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _install_command(foundation_root: Path, venv_path: Path) -> list[str]:
    requirements = _foundation_requirements_path(foundation_root)
    if requirements is None:
        requirements = foundation_root / "requirements.txt"
    use_usd_exchange = _should_use_usd_exchange_provider()
    overrides = None
    if use_usd_exchange:
        requirements = _write_usd_exchange_requirements_file(requirements, venv_path)
        overrides = _write_usd_exchange_overrides_file(venv_path)
    python_executable = _venv_python(venv_path)
    install = [
        "uv",
        "venv",
        "--python",
        sys.executable,
        str(venv_path),
        "&&",
        "uv",
        "pip",
        "install",
        "--python",
        str(python_executable),
    ]
    if use_usd_exchange:
        install.append(SIMREADY_USD_EXCHANGE_REQUIREMENT)
        install.extend(["--overrides", str(overrides)])
    install.extend(
        [
            "-r",
            str(requirements),
            *SIMREADY_VALIDATOR_CONSTRAINTS,
        ]
    )
    if not use_usd_exchange:
        install.append(SIMREADY_USD_CORE_REQUIREMENT)
    return install


def _foundation_requirements_path(foundation_root: Path | None) -> Path | None:
    if foundation_root is None:
        return None
    requirements = foundation_root / "requirements.txt"
    if requirements.exists():
        return requirements
    return foundation_root / "nv_core" / "validator_sample" / "requirements.txt"


def _foundation_requirements_sha256(foundation_root: Path | None) -> str | None:
    requirements = _foundation_requirements_path(foundation_root)
    if requirements is None or not requirements.is_file():
        return None
    try:
        return hashlib.sha256(requirements.read_bytes()).hexdigest()
    except OSError:
        return None


def _should_use_usd_exchange_provider() -> bool:
    provider = os.getenv(SIMREADY_USD_PROVIDER_ENV, "auto").strip().lower()
    if provider in {"usd-exchange", "usd_exchange", "exchange"}:
        return True
    if provider in {"usd-core", "usd_core", "core"}:
        return False
    return (
        sys.platform.startswith("linux")
        and platform.machine().lower() == "aarch64"
        and sys.version_info < (3, 13)
    )


def _write_usd_exchange_requirements_file(
    requirements_path: Path, venv_path: Path
) -> Path:
    filtered = venv_path.with_name(f"{venv_path.name}-usd-exchange-requirements.txt")
    filtered.parent.mkdir(parents=True, exist_ok=True)
    if not requirements_path.exists():
        filtered.write_text(
            (
                "# SimReady Foundation requirements file was not present at "
                f"{requirements_path}.\n"
            ),
            encoding="utf-8",
        )
        return filtered
    lines = requirements_path.read_text(encoding="utf-8").splitlines()
    filtered.write_text(
        "\n".join(
            (
                f"# Replaced by {SIMREADY_USD_EXCHANGE_REQUIREMENT}: {line}"
                if _is_usd_core_requirement_line(line)
                else line
            )
            for line in lines
        )
        + "\n",
        encoding="utf-8",
    )
    return filtered


def _write_usd_exchange_overrides_file(venv_path: Path) -> Path:
    overrides = venv_path.with_name(f"{venv_path.name}-usd-exchange-overrides.txt")
    overrides.parent.mkdir(parents=True, exist_ok=True)
    overrides.write_text(
        (
            "# Disable the usd-core dependency for this interpreter; "
            "usd-exchange provides the pxr runtime.\n"
            f"{SIMREADY_USD_CORE_OVERRIDE}\n"
        ),
        encoding="utf-8",
    )
    return overrides


def _is_usd_core_requirement_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return False
    return bool(re.match(r"(?i)^usd-core(?=$|\s|[<>=!~;#\[])", stripped))


def _prepare_validation_venv(
    command: list[str],
    *,
    expected_marker_identity: dict[str, object],
) -> str | None:
    if not command:
        return "No SimReady validation venv install command was built."
    if "&&" not in command:
        return "Malformed SimReady validation venv install command."
    uv_executable = shutil.which("uv")
    if uv_executable is None:
        return (
            "uv executable is required to prepare the SimReady validation venv. "
            "Install uv or provide CONTENT_WORKFLOW_SIMREADY_VENV."
        )
    split_at = command.index("&&")
    first = command[:split_at]
    second = command[split_at + 1 :]
    uv_path = Path(uv_executable).resolve()
    first[0] = str(uv_path)
    second[0] = str(uv_path)
    venv_path = Path(first[-1]).expanduser().resolve()
    tool_home = venv_path.with_name(f".{venv_path.name}-tool-home")
    subprocess_environment = build_simready_subprocess_environment(
        executable_dir=uv_path.parent,
        allow_network=True,
        isolated_home=tool_home,
    )
    lock_path, lock_fd, lock_error = _acquire_venv_lock(venv_path)
    if lock_error:
        return lock_error
    try:
        if (
            _validator_from_venv(
                venv_path,
                require_ready_marker=True,
                expected_marker_identity=expected_marker_identity,
            )
            is not None
        ):
            return None
        try:
            _venv_ready_marker(venv_path).unlink()
        except FileNotFoundError:
            pass
        shutil.rmtree(venv_path, ignore_errors=True)
        first_completed = subprocess.run(
            first,
            check=False,
            capture_output=True,
            cwd=venv_path.parent,
            env=subprocess_environment,
            text=True,
            timeout=300,
        )
        if first_completed.returncode != 0:
            shutil.rmtree(venv_path, ignore_errors=True)
            return (
                "Failed to create SimReady validation venv: "
                + first_completed.stderr.strip()
            )
        second_completed = subprocess.run(
            second,
            check=False,
            capture_output=True,
            cwd=venv_path.parent,
            env=subprocess_environment,
            text=True,
            timeout=900,
        )
        if second_completed.returncode != 0:
            shutil.rmtree(venv_path, ignore_errors=True)
            return (
                "Failed to install SimReady validation dependencies: "
                + second_completed.stderr.strip()
            )
        if _validator_from_venv(venv_path) is None:
            shutil.rmtree(venv_path, ignore_errors=True)
            return (
                "Failed to install SimReady validation dependencies: "
                "simready-validate executable was not created."
            )
        _write_venv_ready_marker(
            venv_path,
            expected_marker_identity=expected_marker_identity,
        )
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        shutil.rmtree(venv_path, ignore_errors=True)
        return f"Failed to prepare SimReady validation venv: {exc}"
    finally:
        if lock_path is not None and lock_fd is not None:
            _release_venv_lock(lock_path, lock_fd)
    return None


def _acquire_venv_lock(
    venv_path: Path, *, timeout_s: float = 600.0
) -> tuple[Path | None, int | None, str | None]:
    lock_path = venv_path.with_name(f"{venv_path.name}.lock")
    return _acquire_pid_lock(
        lock_path,
        timeout_s=timeout_s,
        label="SimReady validation venv",
    )


def _acquire_foundation_lock(
    root: Path, *, timeout_s: float = 600.0
) -> tuple[Path | None, int | None, str | None]:
    lock_path = root.with_name(f"{root.name}.lock")
    return _acquire_pid_lock(
        lock_path,
        timeout_s=timeout_s,
        label="SimReady Foundation checkout",
    )


def _acquire_pid_lock(
    lock_path: Path,
    *,
    timeout_s: float,
    label: str,
) -> tuple[Path | None, int | None, str | None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_s
    open_flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        open_flags |= os.O_NOFOLLOW
    try:
        lock_fd = os.open(lock_path, open_flags, 0o600)
    except OSError as exc:
        return None, None, f"Failed to open {label} lock: {exc}"
    metadata = os.fstat(lock_fd)
    owner_mismatch = hasattr(os, "geteuid") and metadata.st_uid != os.geteuid()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1 or owner_mismatch:
        os.close(lock_fd)
        return None, None, f"Refusing unsafe {label} lock file: {lock_path}"
    if hasattr(os, "fchmod"):
        try:
            os.fchmod(lock_fd, 0o600)
        except OSError as exc:
            os.close(lock_fd)
            return None, None, f"Failed to secure {label} lock: {exc}"
    while True:
        try:
            if _try_os_file_lock(lock_fd):
                pid_payload = str(os.getpid()).encode("ascii")
                if sys.platform == "win32":
                    os.lseek(lock_fd, 1, os.SEEK_SET)
                    os.write(lock_fd, pid_payload.ljust(32, b" "))
                else:
                    os.ftruncate(lock_fd, 0)
                    os.lseek(lock_fd, 0, os.SEEK_SET)
                    os.write(lock_fd, pid_payload)
                os.fsync(lock_fd)
                return lock_path, lock_fd, None
        except OSError as exc:
            os.close(lock_fd)
            return None, None, f"Failed to acquire {label} lock: {exc}"
        if time.monotonic() >= deadline:
            os.close(lock_fd)
            return (
                None,
                None,
                f"Timed out waiting for {label} lock: {lock_path}",
            )
        time.sleep(0.25)


def _try_os_file_lock(lock_fd: int) -> bool:
    if sys.platform == "win32":
        import msvcrt

        os.lseek(lock_fd, 0, os.SEEK_SET)
        if os.fstat(lock_fd).st_size == 0:
            os.write(lock_fd, b"\0")
            os.fsync(lock_fd)
        os.lseek(lock_fd, 0, os.SEEK_SET)
        try:
            msvcrt.locking(lock_fd, msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True

    import fcntl

    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    return True


def _release_pid_lock(lock_path: Path, lock_fd: int) -> None:
    del lock_path
    try:
        if sys.platform == "win32":
            import msvcrt

            os.lseek(lock_fd, 0, os.SEEK_SET)
            msvcrt.locking(lock_fd, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock_fd, fcntl.LOCK_UN)
    except OSError:
        pass
    try:
        os.close(lock_fd)
    except OSError:
        pass


def _release_venv_lock(lock_path: Path, lock_fd: int) -> None:
    _release_pid_lock(lock_path, lock_fd)


def _venv_python(venv_path: Path) -> Path:
    scripts_dir = "Scripts" if sys.platform == "win32" else "bin"
    executable = "python.exe" if sys.platform == "win32" else "python"
    return venv_path / scripts_dir / executable


def _clone_foundation(root: Path, *, ref: str) -> str | None:
    if invalid_ref := _invalid_git_ref(ref):
        return invalid_ref
    git_executable = _managed_git_executable()
    if git_executable is None:
        return "git is required to clone the managed SimReady Foundation checkout."
    environment = _managed_git_environment(git_executable, allow_network=True)
    root.parent.mkdir(parents=True, exist_ok=True)
    command = _managed_git_command(
        git_executable,
        "clone",
        "--depth",
        "1",
        DEFAULT_SIMREADY_FOUNDATION_REPO_URL,
        str(root),
    )
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            env=environment,
            text=True,
            timeout=900,
        )
        if completed.returncode != 0:
            cleanup_error = _remove_managed_tree(root)
            detail = "Failed to clone SimReady Foundation: " + completed.stderr.strip()
            return f"{detail}; {cleanup_error}" if cleanup_error else detail
        fetch_error = _fetch_foundation_ref(
            root,
            ref=ref,
            git_executable=git_executable,
        )
        if fetch_error:
            cleanup_error = _remove_managed_tree(root)
            return f"{fetch_error}; {cleanup_error}" if cleanup_error else fetch_error
        checkout = subprocess.run(
            _managed_git_command(
                git_executable,
                "-C",
                str(root),
                "checkout",
                "--detach",
                "FETCH_HEAD",
            ),
            check=False,
            capture_output=True,
            env=environment,
            text=True,
            timeout=120,
        )
        if checkout.returncode != 0:
            cleanup_error = _remove_managed_tree(root)
            detail = (
                "Failed to checkout SimReady Foundation ref "
                f"{ref!r}: {checkout.stderr.strip()}"
            )
            return f"{detail}; {cleanup_error}" if cleanup_error else detail
    except (OSError, subprocess.TimeoutExpired) as exc:
        cleanup_error = _remove_managed_tree(root)
        detail = f"Failed to clone SimReady Foundation: {exc}"
        return f"{detail}; {cleanup_error}" if cleanup_error else detail
    return None


def _remove_managed_tree(root: Path) -> str | None:
    """Remove an installer-owned tree, including read-only Git files on Windows."""

    if not root.exists():
        return None

    def make_writable_and_retry(function, path: str, _error: BaseException) -> None:
        os.chmod(path, stat.S_IWRITE)
        function(path)

    try:
        shutil.rmtree(root, onexc=make_writable_and_retry)
    except OSError as exc:
        return f"Failed to remove partial managed checkout {root}: {exc}"
    if root.exists():
        return f"Failed to remove partial managed checkout {root}"
    return None


def _update_foundation(root: Path, *, ref: str) -> str | None:
    if invalid_ref := _invalid_git_ref(ref):
        return invalid_ref
    git_executable = _managed_git_executable()
    if git_executable is None:
        return "git is required to update the managed SimReady Foundation checkout."
    environment = _managed_git_environment(git_executable, allow_network=True)
    try:
        fetch_error = _fetch_foundation_ref(
            root,
            ref=ref,
            git_executable=git_executable,
        )
        if fetch_error:
            return fetch_error
        checkout = subprocess.run(
            _managed_git_command(
                git_executable,
                "-C",
                str(root),
                "checkout",
                "--detach",
                "FETCH_HEAD",
            ),
            check=False,
            capture_output=True,
            env=environment,
            text=True,
            timeout=120,
        )
        if checkout.returncode != 0:
            return (
                "Failed to update SimReady Foundation checkout to "
                f"{ref!r}: {checkout.stderr.strip()}"
            )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"Failed to update SimReady Foundation checkout: {exc}"
    return None


def _invalid_git_ref(ref: str) -> str | None:
    if not ref or ref.startswith("-"):
        return f"Invalid SimReady Foundation ref: {ref!r}"
    return None


def _fetch_foundation_ref(
    root: Path,
    *,
    ref: str,
    git_executable: Path | None = None,
) -> str | None:
    git_executable = git_executable or _managed_git_executable()
    if git_executable is None:
        return "git is required to fetch the managed SimReady Foundation checkout."
    fetch = subprocess.run(
        _managed_git_command(
            git_executable,
            "-C",
            str(root),
            "fetch",
            "--depth",
            "1",
            "origin",
            ref,
        ),
        check=False,
        capture_output=True,
        env=_managed_git_environment(git_executable, allow_network=True),
        text=True,
        timeout=300,
    )
    if fetch.returncode != 0:
        return (
            f"Failed to fetch SimReady Foundation ref {ref!r}: {fetch.stderr.strip()}"
        )
    return None


def _foundation_commit(root: Path | None) -> str | None:
    if root is None or not root.exists():
        return None
    git_executable = _managed_git_executable()
    if git_executable is None:
        return None
    try:
        completed = subprocess.run(
            _managed_git_command(
                git_executable,
                "-C",
                str(root),
                "rev-parse",
                "HEAD",
            ),
            check=False,
            capture_output=True,
            env=_managed_git_environment(git_executable),
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def _managed_git_executable() -> Path | None:
    return _trusted_system_executable(_TRUSTED_GIT_EXECUTABLE_CANDIDATES)


def _managed_git_lfs_executable() -> Path | None:
    return _trusted_system_executable(_TRUSTED_GIT_LFS_EXECUTABLE_CANDIDATES)


def _trusted_system_executable(candidates: tuple[Path, ...]) -> Path | None:
    for candidate in candidates:
        if not candidate.is_absolute():
            continue
        try:
            executable = candidate.resolve(strict=True)
            metadata = executable.stat()
        except OSError:
            continue
        if not stat.S_ISREG(metadata.st_mode) or not os.access(executable, os.X_OK):
            continue
        if os.name != "nt" and (
            metadata.st_uid != 0 or metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
        ):
            continue
        return executable
    return None


def _managed_git_command(git_executable: Path, *args: str) -> list[str]:
    command = [
        str(git_executable),
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.untrackedCache=false",
        "-c",
        "core.autocrlf=false",
    ]
    if os.name == "nt":
        command.extend(("-c", "core.longpaths=true"))
    git_lfs_executable = _managed_git_lfs_executable()
    if git_lfs_executable is not None:
        quote = subprocess.list2cmdline if os.name == "nt" else shlex.join
        command.extend(
            (
                "-c",
                "filter.lfs.clean="
                + quote([str(git_lfs_executable), "clean", "--", "%f"]),
                "-c",
                "filter.lfs.smudge="
                + quote([str(git_lfs_executable), "smudge", "--", "%f"]),
                "-c",
                "filter.lfs.process="
                + quote([str(git_lfs_executable), "filter-process"]),
                "-c",
                "filter.lfs.required=true",
            )
        )
    command.extend(args)
    return command


def _managed_git_environment(
    git_executable: Path,
    *,
    allow_network: bool = False,
) -> dict[str, str]:
    environment = build_simready_subprocess_environment(
        executable_dir=git_executable.parent,
        allow_network=allow_network,
    )
    git_lfs_executable = _managed_git_lfs_executable()
    if git_lfs_executable is not None:
        environment["PATH"] = os.pathsep.join(
            dict.fromkeys(
                (
                    str(git_executable.parent),
                    str(git_lfs_executable.parent),
                    *environment["PATH"].split(os.pathsep),
                )
            )
        )
    environment.update(
        {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_LFS_SKIP_SMUDGE": "0",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    return environment


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result
