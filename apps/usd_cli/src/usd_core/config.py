# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration loading for the in-tree usd-cli component.

Resolution order, later wins:
    built-in defaults
    -> ~/.config/usd-cli/config.toml (global; legacy ov/3dsc honored)
    -> <project>/.ov/config.toml     (project, found by walking up)
    -> OV_* environment variables
    (-> CLI flags are applied by the caller, on top of this)
"""

from __future__ import annotations

import contextlib
import json
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

STATE_DIR_NAME = ".usd-cli"
#: pre-rename state dirs (newest first) — existing projects keep working
#: without migration across BOTH renames (3dsc -> ov-cli -> usd-cli)
LEGACY_STATE_DIR_NAMES = (".ov", ".3dsc")
CONFIG_NAME = "config.toml"
SERVER_STATE_NAME = "server.json"
RENDER_BACKEND_API_KEYS_ENV = (  # pragma: allowlist secret
    "USD_CLI_RENDER_BACKEND_API_KEYS_JSON"
)
_MAX_RENDER_BACKEND_API_KEYS_BYTES = 64 * 1024

DEFAULTS: dict = {
    "backend": {"engine": "usd"},
    "server": {"host": "127.0.0.1", "port": 0, "idle_timeout": "30m"},
    # Select local or remote OVRTX transport; "auto" probes local first, then remote.
    "render": {
        # Match Content Workbench's inspection default: RT2 with 64 updates.
        # `fast` remains available as an explicit RT1 preview mode.
        "mode": "quality",
        "renderer": "auto",
        "remote_url": "",
        "remote_api_key": "",
        # multi-backend pool: [[render.backends]] array of {url, api_key} tables.
        # Non-empty → it IS the pool (deduped by url) and renders are scheduled/sharded
        # across it; empty → the single remote_url/remote_api_key above form a one-entry
        # pool. See resolve_render_backends(). Not settable via OV_* env vars (an
        # array of tables has no env encoding).
        "backends": [],
        "remote_timeout": 300,
        "remote_compress": True,  # gzip the USDZ upload when it helps
        # client-side upload cap (MB); 0 = unlimited. Bounded by default: an unlimited
        # cap let a 1,377 MB bundle tie up the render service for minutes.
        "remote_max_upload_mb": 512,
        # cache packaged USDZ bundles (and their gzip) per content fingerprint in the
        # per-user staging root, so an unchanged scene is never re-packaged between
        # renders (an identical 1.38 GB bundle was packaged 4× in 15 minutes). Keeps
        # the newest 4 bundles per scene, LRU-evicted. false = package every render.
        "remote_bundle_cache": True,
        # Content-addressed upload dedup:
        # when a backend advertises the "cas" feature, ship the bundle as a per-file
        # sha256 manifest and upload only the blobs the backend is missing — a
        # re-render after a small edit uploads KBs, not the scene. false = always
        # ship the whole bundle (the pre-Phase-2 transport).
        "remote_cas": True,
        # refuse remote render/physics when the backend was built from a different
        # checkout (protocol_version mismatch on /live) — see usd_core.remote_protocol
        "remote_verify_version": True,
        "resolution": "1024x1024",
        # ovrtx-only knobs (ignored by other backends):
        # allow pip to download the ~2.5 GB ovrtx wheel at render time when the venv is
        # missing; false = fail fast with install instructions instead of a silent stall
        "ovrtx_auto_install": False,
        "ovrtx_num_sensor_updates": 64,  # path-tracer step iterations (quality knob)
        "ovrtx_render_mode": "",  # "" = derive from fast/quality; or pin rt1/rt2/pt
        "ovrtx_log_level": "warn",
    },
    "session": {},
}


@dataclass
class Config:
    backend: dict = field(default_factory=lambda: dict(DEFAULTS["backend"]))
    server: dict = field(default_factory=lambda: dict(DEFAULTS["server"]))
    render: dict = field(default_factory=lambda: dict(DEFAULTS["render"]))
    session: dict = field(default_factory=lambda: dict(DEFAULTS["session"]))
    #: directory the project config was found in (where server.json will live)
    project_dir: Path | None = None

    @property
    def state_dir(self) -> Path:
        return state_dir_for(self.project_dir or Path.cwd())

    @property
    def server_state_path(self) -> Path:
        return self.state_dir / SERVER_STATE_NAME

    @property
    def checkpoints_dir(self) -> Path:
        """Where `checkpoint save` writes layers (state-model.md §4)."""
        return self.state_dir / "checkpoints"


def state_dir_for(base: Path) -> Path:
    """THE state directory for a project at `base` — the single resolver every
    component must use (config load, render hot-reload, remote configure, daemon
    pids/locks/spawn). A project initialized before a rename keeps its `.ov/`
    (or `.3dsc/`) dir — its config, checkpoints, and server state stay valid
    without migration; new projects (no dir exists) get `.usd-cli/`. When
    several exist (e.g. a tool accidentally mkdir'd an empty `.usd-cli` beside
    live legacy state), the newest-named dir holding real state wins, then the
    newest-named existing dir — so a legacy project's config/checkpoints are
    never shadowed by an empty new-name dir. Resolving this in ONE place is
    what prevents a split-brain project where discovery selects one dir while
    spawn/config create a parallel other."""
    base = Path(base)
    candidates = [base / STATE_DIR_NAME] + [base / n for n in LEGACY_STATE_DIR_NAMES]

    def _has_state(d: Path) -> bool:
        return any(
            (d / n).exists()
            for n in (CONFIG_NAME, SERVER_STATE_NAME, "daemon.pids", "checkpoints")
        )

    existing = [d for d in candidates if d.is_dir()]
    for d in existing:  # newest-named first: real state wins
        if _has_state(d):
            return d
    if existing:  # all empty: prefer the newest-named existing dir
        return existing[0]
    return candidates[0]


def find_project_dir(start: Path | None = None) -> Path | None:
    """Walk up from `start` looking for a `.ov/` (or pre-rename `.3dsc/`) directory."""
    cur = (start or Path.cwd()).resolve()
    for d in (cur, *cur.parents):
        if (d / STATE_DIR_NAME).is_dir() or any(
            (d / n).is_dir() for n in LEGACY_STATE_DIR_NAMES
        ):
            return d
    return None


def _read_toml(path: Path) -> dict:
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except FileNotFoundError:
        return {}
    except tomllib.TOMLDecodeError as exc:  # pragma: no cover - surfaced to user
        raise ValueError(f"invalid TOML in {path}: {exc}") from exc


def _deep_merge(base: dict, overlay: dict) -> dict:
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _validate_remote_verify_version_setting(render: dict) -> None:
    """Reject two easy-to-miss spellings of the emergency protocol override.

    The protocol error used to recommend the dotted
    ``render.remote_verify_version`` name. Pasting that literal text below an
    existing ``[render]`` header is valid TOML, but it means
    ``render.render.remote_verify_version`` and leaves the real setting at its
    strict default. A user report also exposed the nearby ``vesion`` typo. Both
    shapes must fail at config load instead of silently pretending the safety
    override was applied.
    """
    nested = render.get("render")
    if isinstance(nested, dict) and any(
        key in nested for key in ("remote_verify_version", "remote_verify_vesion")
    ):
        raise ValueError(
            "invalid render version override: inside an existing [render] section, "
            "set `remote_verify_version = false` directly (without the `render.` "
            "prefix)"
        )
    if "remote_verify_vesion" in render:
        raise ValueError(
            "unknown render setting 'remote_verify_vesion'; did you mean "
            "'remote_verify_version'?"
        )


def _env_overrides() -> dict:
    """Map USD_CLI_<SECTION>_<KEY> env vars onto the config tree.

    Pre-rename `3DSC_*` and `OV_*` variables are merged FIRST (an environment
    set up for an older name keeps working); newer-name values win on
    conflict, `USD_CLI_*` strongest."""
    out: dict = {}
    for prefix in ("3DSC_", "OV_", "USD_CLI_"):
        for env_key, val in os.environ.items():
            if not env_key.startswith(prefix):
                continue
            parts = env_key[len(prefix) :].lower().split("_", 1)
            if len(parts) != 2:
                continue
            section, key = parts
            out.setdefault(section, {})[key] = val
    return out


def env_get(name: str) -> str | None:
    """A USD_CLI_* env var, falling back to its pre-rename OV_* / 3DSC_* twins
    — the ONE lookup non-config controls (NO_DAEMON, DEBUG, tokens) share."""
    assert name.startswith("USD_CLI_"), name
    suffix = name[len("USD_CLI_") :]
    return (
        os.environ.get(name)
        or os.environ.get("OV_" + suffix)
        or os.environ.get("3DSC_" + suffix)
    )


def resolve_render_backends(render: dict) -> list[dict]:
    """The remote render backend pool: [{"url", "api_key"}], deduped by URL.

    `[[render.backends]]` entries win when present; otherwise the legacy single
    `remote_url`/`remote_api_key` fields form a one-entry pool. Entries without a
    url are skipped; urls are normalized (trailing slash stripped) before dedup —
    first occurrence wins. Empty list = no remote backend configured at all.
    """
    pool: list[dict] = []
    seen: set[str] = set()
    entries = render.get("backends") or []
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            url = str(entry.get("url", "") or "").rstrip("/")
            if not url or url in seen:
                continue
            seen.add(url)
            pool.append({"url": url, "api_key": str(entry.get("api_key", "") or "")})
    if pool:
        return pool
    url = str(render.get("remote_url", "") or "").rstrip("/")
    if url:
        return [{"url": url, "api_key": str(render.get("remote_api_key", "") or "")}]
    return []


def _toml_value(v: object) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    s = str(v).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{s}"'


def dump_toml(data: dict) -> str:
    """Serialize a config dict: top-level scalars, one level of named tables, plain
    arrays, and arrays of tables one level under a table (`[[section.key]]`) — the
    shapes a usd-cli config.toml uses (`[[render.backends]]` is the array-of-tables
    case). Existing keys are preserved; comments are not (tomllib can't round-trip
    them)."""
    scalars = {k: v for k, v in data.items() if not isinstance(v, (dict, list))}
    tables = {k: v for k, v in data.items() if isinstance(v, dict)}
    lines: list[str] = [f"{k} = {_toml_value(v)}" for k, v in scalars.items()]
    for name, tbl in tables.items():
        if lines:
            lines.append("")
        lines.append(f"[{name}]")
        arrays_of_tables: dict[str, list] = {}
        for k, v in tbl.items():
            if isinstance(v, list) and v and all(isinstance(e, dict) for e in v):
                arrays_of_tables[k] = v  # emitted after the table's scalar keys
            elif isinstance(v, list):
                lines.append(f"{k} = [{', '.join(_toml_value(e) for e in v)}]")
            else:
                lines.append(f"{k} = {_toml_value(v)}")
        for k, entries in arrays_of_tables.items():
            for entry in entries:
                lines.append("")
                lines.append(f"[[{name}.{k}]]")
                lines.extend(f"{ek} = {_toml_value(ev)}" for ek, ev in entry.items())
    return "\n".join(lines) + "\n"


@contextlib.contextmanager
def config_file_lock(path: Path):
    """Cross-process exclusive lock for a config read-modify-write transaction.

    flock(2) on a `.lock` sibling of `path`: two concurrent `remote configure --add`
    calls used to both read the same base config and last-write-wins away one entry.
    The lock file itself is left in place — unlinking a file others may be flocking
    is racy (a third writer could lock the unlinked inode and race the fourth)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            import fcntl  # POSIX only; Windows falls back to no cross-process lock
        except ImportError:
            fcntl = None
        if fcntl is not None:
            fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)  # closing the fd releases the flock


def write_config_atomic(path: Path, text: str) -> None:
    """Publish a config file atomically, never world-readable.

    The old `write_text` + chmod-after flow published API keys with default (0644)
    permissions for a window and could leave a truncated TOML if interrupted.
    Here: write a mode-0600 O_CREAT|O_EXCL temp sibling, fsync it, `os.replace`
    over the target, then fsync the directory so the rename itself is durable."""
    import secrets

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    fd = os.open(str(tmp), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    try:  # persist the rename (best effort: some filesystems refuse directory fsync)
        dfd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except OSError:
        pass


def load_config(start: Path | None = None) -> Config:
    project_dir = find_project_dir(start)
    global_path = Path.home() / ".config" / "usd-cli" / CONFIG_NAME
    if not global_path.exists():
        for legacy in ("ov", "3dsc"):  # pre-rename global configs keep working
            legacy_global = Path.home() / ".config" / legacy / CONFIG_NAME
            if legacy_global.exists():
                global_path = legacy_global
                break

    merged = dict(DEFAULTS)
    merged = _deep_merge(merged, _read_toml(global_path))
    if project_dir is not None:
        merged = _deep_merge(
            merged, _read_toml(state_dir_for(project_dir) / CONFIG_NAME)
        )
    merged = _deep_merge(merged, _env_overrides())
    render = dict(merged.get("render", {}))
    _validate_remote_verify_version_setting(render)
    backend_api_keys_json = render.pop("backend_api_keys_json", None)
    if backend_api_keys_json:
        if (
            not isinstance(backend_api_keys_json, str)
            or len(backend_api_keys_json.encode("utf-8"))
            > _MAX_RENDER_BACKEND_API_KEYS_BYTES
        ):
            raise ValueError("render backend API-key injection is oversized")
        try:
            raw_backend_api_keys = json.loads(backend_api_keys_json)
        except json.JSONDecodeError as exc:
            raise ValueError("render backend API-key injection is invalid JSON") from exc
        if not isinstance(raw_backend_api_keys, dict) or any(
            not isinstance(url, str) or not isinstance(api_key, str)
            for url, api_key in raw_backend_api_keys.items()
        ):
            raise ValueError(
                "render backend API-key injection must map URLs to string keys"
            )
        backend_api_keys = {
            url.rstrip("/"): api_key for url, api_key in raw_backend_api_keys.items()
        }
        backends = render.get("backends")
        if isinstance(backends, list):
            render["backends"] = [
                (
                    {
                        **entry,
                        "api_key": backend_api_keys[str(entry.get("url", "")).rstrip("/")],
                    }
                    if isinstance(entry, dict)
                    and str(entry.get("url", "")).rstrip("/") in backend_api_keys
                    else entry
                )
                for entry in backends
            ]
    merged["render"] = render

    engine = str(merged.get("backend", {}).get("engine", "usd")).lower()
    if engine != "usd":
        raise ValueError(
            f"unsupported backend.engine '{engine}' (only 'usd' is implemented)"
        )
    renderer = str(merged.get("render", {}).get("renderer", "auto")).lower()
    if renderer not in {"auto", "ovrtx", "remote"}:
        raise ValueError(
            f"unsupported render.renderer '{renderer}'; OVRTX is the only "
            "supported rendering engine"
        )

    return Config(
        backend=merged.get("backend", {}),
        server=merged.get("server", {}),
        render=merged.get("render", {}),
        session=merged.get("session", {}),
        project_dir=project_dir,
    )
