# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime discovery for SimReady Foundation workflow adapters."""

from __future__ import annotations

import hashlib
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import tomllib
from pathlib import Path

from .models import (
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
SIMREADY_USD_EXCHANGE_REQUIREMENT = "usd-exchange>=2.3,<3"
SIMREADY_USD_CORE_EXCLUDE = "usd-core"


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

    if root is not None and managed and (install_missing or update_foundation):
        lock_path, lock_fd, lock_error = _acquire_foundation_lock(root)
        if lock_error:
            errors.append(lock_error)
        else:
            try:
                if not root.exists() and install_missing:
                    clone_error = _clone_foundation(root, ref=ref)
                    if clone_error:
                        errors.append(clone_error)
                elif root.exists() and update_foundation:
                    update_error = _update_foundation(root, ref=ref)
                    if update_error:
                        warnings.append(update_error)
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
    spec_root = _resolve_spec_root(root, foundation_spec_root)
    specs_ready, spec_errors = _check_spec_root(spec_root)
    errors.extend(spec_errors)

    selected_venv, managed_venv = _resolve_venv_path(venv_path, root)
    validator = _validator_from_venv(
        selected_venv,
        require_ready_marker=managed_venv,
    )
    install_command: list[str] = []

    if validator is None and install_missing and root is not None and root.exists():
        install_command = _install_command(root, selected_venv)
        install_error = _prepare_validation_venv(install_command)
        if install_error:
            errors.append(install_error)
        validator = _validator_from_venv(
            selected_venv,
            require_ready_marker=True,
        )

    if validator is None:
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

    return SimReadyRuntimeInfo(
        foundation_repo_url=DEFAULT_SIMREADY_FOUNDATION_REPO_URL,
        foundation_ref=ref,
        foundation_root=str(root) if root is not None else None,
        foundation_commit=commit,
        foundation_spec_root=str(spec_root) if spec_root is not None else None,
        managed_foundation_checkout=managed,
        venv_path=str(selected_venv) if selected_venv is not None else None,
        validator_executable=str(validator) if validator is not None else None,
        install_command=install_command,
        available_profiles=profiles,
        specs_ready=specs_ready,
        runtime_ready=runtime_ready,
        warnings=_dedupe(warnings),
        errors=_dedupe(errors),
    )


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
    return [
        runtime.validator_executable,
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
    venv_path: Path | str | None, foundation_root: Path | None
) -> tuple[Path, bool]:
    if venv_path is not None:
        return Path(venv_path).expanduser().resolve(), False
    env_venv = os.getenv(SIMREADY_VENV_ENV)
    if env_venv:
        return Path(env_venv).expanduser().resolve(), False
    root_key = str(foundation_root) if foundation_root is not None else "unresolved"
    provider_key = "usd-exchange" if _should_use_usd_exchange_provider() else "usd-core"
    digest = hashlib.sha256(f"{root_key}|provider={provider_key}".encode()).hexdigest()[
        :12
    ]
    return _cache_dir() / "venvs" / f"simready-foundation-{digest}", True


def _cache_dir() -> Path:
    configured = os.getenv(SIMREADY_CACHE_DIR_ENV)
    if configured:
        return Path(configured).expanduser().resolve()
    base = Path(os.getenv("XDG_CACHE_HOME", "~/.cache")).expanduser()
    return base / "content-agent-workflows" / "simready"


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-")
    return safe or "main"


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
) -> Path | None:
    if venv_path is None:
        return None
    if require_ready_marker and not _venv_ready_marker(venv_path).exists():
        return None
    executable = (
        "simready-validate.exe" if sys.platform == "win32" else "simready-validate"
    )
    scripts_dir = "Scripts" if sys.platform == "win32" else "bin"
    candidate = venv_path / scripts_dir / executable
    return candidate if candidate.exists() else None


def _venv_ready_marker(venv_path: Path) -> Path:
    return venv_path / SIMREADY_VENV_READY_MARKER


def _install_command(foundation_root: Path, venv_path: Path) -> list[str]:
    requirements = foundation_root / "requirements.txt"
    if not requirements.exists():
        requirements = (
            foundation_root / "nv_core" / "validator_sample" / "requirements.txt"
        )
    use_usd_exchange = _should_use_usd_exchange_provider()
    excludes = None
    if use_usd_exchange:
        requirements = _write_usd_exchange_requirements_file(requirements, venv_path)
        excludes = _write_usd_exchange_excludes_file(venv_path)
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
        install.extend(["--excludes", str(excludes)])
    install.extend(
        [
            "-r",
            str(requirements),
        ]
    )
    return install


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


def _write_usd_exchange_excludes_file(venv_path: Path) -> Path:
    excludes = venv_path.with_name(f"{venv_path.name}-usd-exchange-excludes.txt")
    excludes.parent.mkdir(parents=True, exist_ok=True)
    excludes.write_text(
        (
            "# Exclude usd-core from direct and transitive Foundation "
            "dependency resolution; usd-exchange provides the pxr runtime.\n"
            f"{SIMREADY_USD_CORE_EXCLUDE}\n"
        ),
        encoding="utf-8",
    )
    return excludes


def _is_usd_core_requirement_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return False
    return bool(re.match(r"(?i)^usd-core(?=$|\s|[<>=!~;#\[])", stripped))


def _prepare_validation_venv(command: list[str]) -> str | None:
    if not command:
        return "No SimReady validation venv install command was built."
    if "&&" not in command:
        return "Malformed SimReady validation venv install command."
    if shutil.which("uv") is None:
        return (
            "uv executable is required to prepare the SimReady validation venv. "
            "Install uv or provide CONTENT_WORKFLOW_SIMREADY_VENV."
        )
    split_at = command.index("&&")
    first = command[:split_at]
    second = command[split_at + 1 :]
    venv_path = Path(first[-1]).expanduser().resolve()
    lock_path, lock_fd, lock_error = _acquire_venv_lock(venv_path)
    if lock_error:
        return lock_error
    try:
        if _validator_from_venv(venv_path, require_ready_marker=True) is not None:
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
        _venv_ready_marker(venv_path).write_text("ready\n", encoding="utf-8")
    except (OSError, subprocess.TimeoutExpired) as exc:
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
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            os.write(fd, str(os.getpid()).encode("utf-8"))
            return lock_path, fd, None
        except FileExistsError:
            if not _venv_lock_holder_active(lock_path):
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass
                except OSError:
                    pass
                else:
                    continue
            if time.monotonic() >= deadline:
                return (
                    None,
                    None,
                    f"Timed out waiting for {label} lock: {lock_path}",
                )
            time.sleep(0.25)
        except OSError as exc:
            return None, None, f"Failed to acquire {label} lock: {exc}"


def _venv_lock_holder_active(lock_path: Path) -> bool:
    try:
        text = lock_path.read_text(encoding="utf-8").strip()
        pid = int(text)
    except (OSError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        if sys.platform == "win32" and getattr(exc, "winerror", None) == 87:
            return False
        return True
    return True


def _release_pid_lock(lock_path: Path, lock_fd: int) -> None:
    try:
        os.close(lock_fd)
    except OSError:
        pass
    try:
        lock_path.unlink()
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
    root.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "git",
        "clone",
        "--depth",
        "1",
        DEFAULT_SIMREADY_FOUNDATION_REPO_URL,
        str(root),
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=900,
        )
        if completed.returncode != 0:
            shutil.rmtree(root, ignore_errors=True)
            return "Failed to clone SimReady Foundation: " + completed.stderr.strip()
        if ref != DEFAULT_SIMREADY_FOUNDATION_REF:
            fetch_error = _fetch_foundation_ref(root, ref=ref)
            if fetch_error:
                shutil.rmtree(root, ignore_errors=True)
                return fetch_error
            checkout = subprocess.run(
                ["git", "-C", str(root), "checkout", "--detach", "FETCH_HEAD"],
                check=False,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if checkout.returncode != 0:
                shutil.rmtree(root, ignore_errors=True)
                return (
                    "Failed to checkout SimReady Foundation ref "
                    f"{ref!r}: {checkout.stderr.strip()}"
                )
    except (OSError, subprocess.TimeoutExpired) as exc:
        shutil.rmtree(root, ignore_errors=True)
        return f"Failed to clone SimReady Foundation: {exc}"
    return None


def _update_foundation(root: Path, *, ref: str) -> str | None:
    if invalid_ref := _invalid_git_ref(ref):
        return invalid_ref
    try:
        fetch_error = _fetch_foundation_ref(root, ref=ref)
        if fetch_error:
            return fetch_error
        checkout = subprocess.run(
            ["git", "-C", str(root), "checkout", "--detach", "FETCH_HEAD"],
            check=False,
            capture_output=True,
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


def _fetch_foundation_ref(root: Path, *, ref: str) -> str | None:
    fetch = subprocess.run(
        ["git", "-C", str(root), "fetch", "--depth", "1", "origin", ref],
        check=False,
        capture_output=True,
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
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


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
