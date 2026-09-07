# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""usd-cli — the command surface (cli-design.md).

Thin Typer app: each command builds a compact payload and hands it to the client. No
scene logic lives here. Reserved-word commands (`import`, `set`, `eval`) use a trailing
underscore on the function with an explicit command name.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import typer

from usd_cli.client import dispatch
from usd_cli.output import EXIT_RUNTIME, emit
from usd_cli.parsing import drop_none, resolution, signed, vec
from usd_cli.state import G

_EXTERNAL_LIFECYCLE_ENV = "USD_CLI_LIFECYCLE_EXTERNALLY_OWNED"
_EXTERNAL_LIFECYCLE_BOOTSTRAP_ENV = "USD_CLI_EXTERNAL_LIFECYCLE_BOOTSTRAP"
_PARENT_SERVER_URL_ENV = "CONTENT_WORKFLOW_USD_CLI_SERVER_URL"
_PARENT_SESSION_ID_ENV = "CONTENT_WORKFLOW_USD_CLI_SESSION_ID"
_LOCAL_GPU_FORBIDDEN_ENV = "USD_CLI_LOCAL_GPU_FORBIDDEN"
_RENDERER_PROCESS_COMMANDS = frozenset(
    {"render", "render-frames", "physics.simulate"}
)


def _configure_windows_text_streams() -> None:
    """Keep Unicode help and diagnostics reliable under legacy Windows code pages."""

    if os.name != "nt":
        return
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="backslashreplace")


# Agents pay tokens for every help screen: Rich's box-drawing help was 29% of ALL
# paid tool output in the round-7 trace audit (459 screens, ~1 MB of borders). A
# non-TTY consumer (or USD_CLI_AGENT=1) gets click's plain compact help instead; humans
# in a terminal keep the pretty version. `usd-cli cheatsheet` is the cheaper entry point.
_PLAIN_HELP = (not sys.stdout.isatty()) or bool(
    os.environ.get("USD_CLI_AGENT") or os.environ.get("OV_AGENT")
    or os.environ.get("3DSC_AGENT"))

app = typer.Typer(
    # no rich traceback boxes: a mistyped `--screen 512 512` printed a 40-line
    # panel; parse errors exit cleanly via run()'s ValueError handler instead
    pretty_exceptions_enable=False,
    name="usd-cli",
    help="usd-cli — perceive and manipulate USD scenes over a token-efficient CLI. "
         "Run `usd-cli cheatsheet` for a one-screen tour of every verb.",
    no_args_is_help=True,
    add_completion=True,
    rich_markup_mode=None if _PLAIN_HELP else "rich",
)


def _run(command: str, payload: dict, *, preserve_false: tuple[str, ...] = ()) -> None:
    """Dispatch a command and exit with the response's status code."""
    body = drop_none(payload)
    # Most false-valued CLI flags mean "leave the server default alone". A dual
    # boolean option whose server default is True is different: `--no-smooth` must
    # cross the boundary as an explicit False rather than disappearing in drop_none.
    for key in preserve_false:
        if payload.get(key) is False:
            body[key] = False
    if (
        not G.server
        and os.environ.get(_LOCAL_GPU_FORBIDDEN_ENV) == "1"
        and command in _RENDERER_PROCESS_COMMANDS
    ):
        from usd_core.models import Issue, Response

        response = Response(
            command=command,
            ok=False,
            summary={},
            data={"error_type": "parent_renderer_capability_required"},
            issues=[
                Issue(
                    "error",
                    "local renderer startup is forbidden in this process; use "
                    "the parent-owned usd-cli capability",
                )
            ],
        )
        raise typer.Exit(emit(response))
    raise typer.Exit(emit(dispatch(command, body)))


def _version_cb(value: bool) -> None:
    if value:
        from importlib.metadata import PackageNotFoundError, version

        from usd_core import load_config

        from usd_core.render.factory import resolved_renderer
        cfg = load_config()
        try:
            ver = version("usd-cli")
        except PackageNotFoundError:  # source checkout without an install
            ver = "0+unknown"
        typer.echo(f"usd-cli {ver} (engine: {cfg.backend.get('engine')}, "
                   f"renderer: {resolved_renderer(cfg)})")
        raise typer.Exit()


@app.callback()
def main_callback(
    version: bool = typer.Option(None, "--version", callback=_version_cb, is_eager=True,
                                 help="Show version + active backend."),
    json_: bool = typer.Option(False, "--json", help="Structured output with schema_version."),
    quiet: bool = typer.Option(False, "-q", "--quiet", help="Payload only; suppress summary lines."),
    server: Optional[str] = typer.Option(None, "--server", help="Daemon URL (default: auto-discover)."),
    session: Optional[str] = typer.Option(
        None,
        "--session",
        help="Named session (default: USD_CLI_SESSION, then the shared default).",
    ),
    timeout: float = typer.Option(
        -1.0, "--timeout",
        help="Total wall-time ceiling in seconds. When set, bounds even renders/physics "
             "(default: 30s for quick commands, 1800s for render/physics)."),
) -> None:
    G.json, G.quiet = json_, quiet
    G.server = server or os.environ.get(_PARENT_SERVER_URL_ENV) or None
    G.session = (
        session
        or os.environ.get(_PARENT_SESSION_ID_ENV)
        or os.environ.get("USD_CLI_SESSION")
        or None
    )
    G.timeout_explicit = timeout >= 0
    G.timeout = timeout if timeout >= 0 else 30.0


_CHEATSHEET = """\
usd-cli cheatsheet — every verb, one line each. Details: usd-cli <verb> --help. Global flags
(BEFORE the verb): --json  -q(uiet)  --session NAME  --timeout S  --server URL

PERCEIVE
  open FILE [--read-only] [--force-reload]   load a stage (daemon auto-starts)
  snapshot [@ref|PATH] [-d N] [-m] [-b] [-p] compact @ref tree; -D diff vs last,
           [--structural] [--since CKPT]      --since diffs vs a checkpoint
  find [--name G] [--type T] [--material M] [--has-attr A]\n       [--where 'name~=G*'] [--under @ref] [--count]   search prims\n  bounds [@ref] | visibility @ref            world bbox / why is it hidden?
  properties @ref [ATTR...]                  attributes of one prim (ATTRs: full values)
  describe [@ref] | info | sublayers         NL summary / stage banner / layers
  resolve [@ref ...]                         reveal full paths (omit: whole table)
  render [-p] [--depth|--normals|--seg|-w] [--focus @ref] [--orbit N] [--res WxH]
         [--against IMG [--diff visual|geometry]] [--max-upload-mb N]
  render-frames [--frames 0:30] [--scene F]  animation frames + GIF
  render|save|checkpoint load --detach       -> job id; usd-cli wait [JOB] — NO poll loops
  jobs | cancel JOB                          list or cooperatively cancel detached jobs
  camera list|use|create|look-at|orbit|fit|pan|zoom|coverage|place|rig-export
  distance|nearest|within|overlapping|above|below|left|right|front|behind|raycast

ACT
  transform @ref --tx=+3 --ry 90 [--scale]   move/rotate/scale (+/-= relative)
  material @ref --color R,G,B | --name N | --library FILE --name N | --bind PATH
  material --type Mesh --under PATH --where 'name~=G*' --bind PATH    bulk rules
  material-apply PLAN.json --library FILE     exact heterogeneous plan, one transaction
  appearance clear                            clean-slate session overlay (undoable)
  create|delete|duplicate|reparent|rename|group|import|align|scatter
  show|hide|isolate|select|set @ref ATTR VALUE|remove-api
  physics apply -f operations.json           Explicit RigidBody/Collision/Mass operations
  validate --fix                             repair dangling bindings etc.

VERIFY
  verify [FILE] [--strict]                   one-shot deliverable check (fresh open);
                                             --strict: warnings (external deps,
                                             hidden renderables) FAIL the verdict
  validate | material audit [--effective]    structural checks / binding coverage
  appearance audit                           effective material/shader/display check
  material-binding @ref                      what's bound + shader params
  physics validate | physics simulate -f request.json          schema / explicit simulation
  snapshot -D                                only what changed

PERSIST + SESSION
  save [FILE] [--flatten]                    verified-atomic; --flatten = self-contained
  export FMT [FILE] | convert SRC [OUT]      exports / converter
  checkpoint save|load|list|delete NAME      named stage checkpoints
  undo | redo | history                      op-log
  server status|stop|restart | remote configure|backends [--deep]

RULES OF THUMB: refs (@n5) beat paths — and they are DETERMINISTIC by prim
path, so two sessions (or a sub-agent) opening the same file state see the
same @refs: share refs in briefs, not long paths. snapshot after edits;
--json only when you must parse; open --read-only for parallel inspection;
busy exit-4 = wait + verify, do NOT restart the daemon.
"""


@app.command()
def cheatsheet():
    """One-screen tour of every verb (the token-cheap alternative to --help crawls)."""
    typer.echo(_CHEATSHEET)
    raise typer.Exit(0)


# ── refs ──────────────────────────────────────────────────────────────────
@app.command()
def resolve(refs: Optional[List[str]] = typer.Argument(None, help="Refs to reveal; omit to list all.")):
    """Reveal the full SdfPath(s) behind ref(s), or list the whole ref table."""
    _run("resolve", {"refs": refs})


# ── snapshot ──────────────────────────────────────────────────────────────
@app.command()
def snapshot(
    scope: Optional[str] = typer.Argument(None, help="Path or @ref to scope to."),
    visible: bool = typer.Option(False, "-v", "--visible"),
    type_: Optional[List[str]] = typer.Option(None, "-t", "--type", help="Filter by prim type (repeatable)."),
    depth: Optional[int] = typer.Option(None, "-d", "--depth"),
    scope_opt: Optional[str] = typer.Option(None, "-s", "--scope"),
    materials: bool = typer.Option(False, "-m", "--materials"),
    bounds: bool = typer.Option(False, "-b", "--bounds"),
    properties: bool = typer.Option(False, "-p", "--properties"),
    diff: bool = typer.Option(False, "-D", "--diff", help="Diff vs the previous snapshot."),
    structural: bool = typer.Option(False, "--structural",
                                    help="Attribute-level diff (e.g. @n5 -t(2,0,0) +t(5,0,0)) vs text."),
    since: Optional[str] = typer.Option(None, "--since", help="Diff against a named checkpoint."),
):
    """Compact text tree of the scene. The core perceive step.

    Refs are deterministic by SdfPath and stable within a session, so -D diffs show only
    real changes.
    """
    _run("snapshot", {
        "scope": scope or scope_opt, "visible": visible, "type": type_, "depth": depth,
        "materials": materials, "bounds": bounds, "properties": properties,
        "diff": diff, "structural": structural, "since": since,
    })


# ── render ────────────────────────────────────────────────────────────────
@app.command()
def render(
    photoreal: bool = typer.Option(False, "-p", "--photoreal"),
    depth: bool = typer.Option(False, "--depth"),
    normals: bool = typer.Option(False, "-n", "--normals"),
    seg: bool = typer.Option(False, "--seg"),
    wireframe: bool = typer.Option(False, "-w", "--wireframe"),
    annotate: bool = typer.Option(False, "-a", "--annotate"),
    focus: Optional[str] = typer.Option(None, "--focus", help="Auto-frame this @ref."),
    orbit: Optional[int] = typer.Option(None, "--orbit", help="N viewpoints around target."),
    elevation: Optional[float] = typer.Option(None, "--elevation"),
    res: Optional[str] = typer.Option(None, "--res", help="WxH, e.g. 1920x1080."),
    inline: bool = typer.Option(False, "--inline", help="Also return base64 inline."),
    mode: Optional[str] = typer.Option(
        None, "--mode", help="fast | quality (default: quality/RT2)."
    ),
    renderer: Optional[str] = typer.Option(
        None,
        "--renderer",
        help="auto | ovrtx | remote (both explicit choices use OVRTX).",
    ),
    against: Optional[str] = typer.Option(None, "--against", help="Render-diff vs a prior render's beauty image."),
    diff: Optional[str] = typer.Option(None, "--diff",
                                       help="Diff type for --against: visual | geometry | both. geometry diffs "
                                            "segmentation footprints, catching moved glass/transparent objects."),
    exclude: Optional[List[str]] = typer.Option(None, "--exclude", help="Refs to exclude (repeatable)."),
    output: Optional[str] = typer.Option(None, "-o", "--output"),
    max_upload_mb: Optional[float] = typer.Option(
        None, "--max-upload-mb",
        help="Per-render override of render.remote_max_upload_mb (0 = unlimited) — "
             "for scenes that legitimately exceed the safe default cap."),
    detach: bool = typer.Option(
        False, "--detach",
        help="Return a job id immediately; collect with `usd-cli wait <job>` — no more "
             "poll loops on long renders."),
):
    """Multimodal render — modalities combine in a single pass."""
    _run("render", {
        "photoreal": photoreal, "depth": depth, "normals": normals, "seg": seg,
        "wireframe": wireframe, "annotate": annotate, "focus": focus, "orbit": orbit,
        "elevation": elevation, "res": resolution(res), "inline": inline, "mode": mode,
        "renderer": renderer, "against": against, "diff": diff, "exclude": exclude, "output": output,
        "max_upload_mb": max_upload_mb, "detach": detach,
    })


@app.command("render-frames")
def render_frames(
    frames: Optional[str] = typer.Option(None, "--frames", help="FrameSpec: '0:30' range, '0,5,10' list, or '42'. Default: the stage's full authored animation range (clamped to it)."),
    scene: Optional[str] = typer.Option(None, "--scene", help="USD to play (e.g. a physics recording.usda); default: the open stage."),
    res: Optional[str] = typer.Option(None, "--res", help="WxH."),
    camera: Optional[str] = typer.Option(None, "--camera", help="Camera @ref/path to render through."),
    focus: Optional[str] = typer.Option(None, "--focus", help="Auto-frame this @ref."),
    fps: Optional[float] = typer.Option(None, "--fps", help="GIF frames/sec. Default: the stage's timeCodesPerSecond, else 24."),
    mode: Optional[str] = typer.Option(
        None, "--mode", help="fast | quality (default: quality/RT2)."
    ),
    renderer: Optional[str] = typer.Option(None, "--renderer"),
    no_animate: bool = typer.Option(False, "--no-animate", help="Skip the assembled GIF; keep frames only."),
    output: Optional[str] = typer.Option(None, "-o", "--output"),
):
    """Render a USD over a frame range at its time samples, with an animated GIF."""
    _run("render-frames", {
        "frames": frames, "scene": scene, "res": resolution(res), "camera": camera,
        "focus": focus, "fps": fps, "mode": mode, "renderer": renderer,
        "animate": not no_animate, "output": output,
    })


# ── manipulation ──────────────────────────────────────────────────────────
@app.command()
def transform(
    ref: str = typer.Argument(...),
    tx: Optional[str] = typer.Option(None, "--tx", "--translate-x"),
    ty: Optional[str] = typer.Option(None, "--ty", "--translate-y"),
    tz: Optional[str] = typer.Option(None, "--tz", "--translate-z"),
    rx: Optional[str] = typer.Option(None, "--rx", "--rotate-x"),
    ry: Optional[str] = typer.Option(None, "--ry", "--rotate-y"),
    rz: Optional[str] = typer.Option(None, "--rz", "--rotate-z"),
    sx: Optional[str] = typer.Option(None, "--sx", "--scale-x"),
    sy: Optional[str] = typer.Option(None, "--sy", "--scale-y"),
    sz: Optional[str] = typer.Option(None, "--sz", "--scale-z"),
    translate: Optional[str] = typer.Option(None, "--translate", help="Vector x,y,z."),
    rotate: Optional[str] = typer.Option(None, "--rotate", help="Vector x,y,z (degrees)."),
    scale: Optional[str] = typer.Option(None, "--scale", help="Vector x,y,z."),
):
    """Translate / rotate / scale a prim. +/- = relative, bare = absolute.

    Tip: use '--tx=-3' (with '=') for negative values so the shell doesn't read -3 as a flag.
    """
    _run("transform", {
        "ref": ref,
        "tx": signed(tx), "ty": signed(ty), "tz": signed(tz),
        "rx": signed(rx), "ry": signed(ry), "rz": signed(rz),
        "sx": signed(sx), "sy": signed(sy), "sz": signed(sz),
        "translate": vec(translate, 3), "rotate": vec(rotate, 3), "scale": vec(scale, 3),
    })


@app.command()
def create(
    type_: str = typer.Argument(..., metavar="TYPE", help="mesh | light | camera | xform | material"),
    name: Optional[str] = typer.Argument(None),
    parent: Optional[str] = typer.Option(None, "--parent"),
    at: Optional[str] = typer.Option(None, "--at", help="Position x,y,z."),
    shape: Optional[str] = typer.Option(None, "--shape", help="Primitive shape for meshes."),
):
    """Create a new prim."""
    _run("create", {"type": type_, "name": name, "parent": parent, "at": vec(at, 3), "shape": shape})


@app.command()
def delete(refs: List[str] = typer.Argument(...)):
    """Delete one or more prims."""
    _run("delete", {"refs": refs})


@app.command()
def duplicate(
    ref: str = typer.Argument(...),
    at: Optional[str] = typer.Option(None, "--at"),
    name: Optional[str] = typer.Option(None, "--name"),
):
    """Duplicate a prim."""
    _run("duplicate", {"ref": ref, "at": vec(at, 3), "name": name})


@app.command()
def reparent(ref: str = typer.Argument(...), under: str = typer.Option(..., "--under")):
    """Reparent a prim under another."""
    _run("reparent", {"ref": ref, "under": under})


@app.command()
def material(
    ref: Optional[str] = typer.Argument(
        None, help="Target @ref/path; 'audit' runs the binding audit; omit for bulk selection."),
    color: Optional[str] = typer.Option(None, "--color", help="r,g,b in 0..1."),
    metallic: Optional[float] = typer.Option(None, "--metallic"),
    roughness: Optional[float] = typer.Option(None, "--roughness"),
    opacity: Optional[float] = typer.Option(None, "--opacity"),
    clearcoat: Optional[float] = typer.Option(None, "--clearcoat", help="Clearcoat weight 0..1."),
    clearcoat_roughness: Optional[float] = typer.Option(None, "--clearcoat-roughness"),
    ior: Optional[float] = typer.Option(None, "--ior", help="Index of refraction."),
    emissive: Optional[str] = typer.Option(None, "--emissive", help="Emissive color r,g,b."),
    bind: Optional[str] = typer.Option(None, "--bind", help="Bind existing material @m."),
    unbind: bool = typer.Option(False, "--unbind", help="Remove the prim's direct material binding."),
    omnipbr: bool = typer.Option(False, "--omnipbr", help="Author an MDL OmniPBR material."),
    mdl_module: Optional[str] = typer.Option(
        None, "--mdl", help="MDL module asset, e.g. OmniGlass.mdl (implies MDL)."),
    subidentifier: Optional[str] = typer.Option(
        None, "--mdl-id", help="MDL subIdentifier (defaults to the module stem)."),
    diffuse_texture: Optional[str] = typer.Option(None, "--diffuse-tex", help="Albedo map."),
    normal_texture: Optional[str] = typer.Option(None, "--normal-tex", help="Normal map."),
    orm_texture: Optional[str] = typer.Option(None, "--orm-tex", help="ORM map (MDL)."),
    roughness_texture: Optional[str] = typer.Option(None, "--roughness-tex",
                                                    help="Roughness map (preview surface)."),
    metallic_texture: Optional[str] = typer.Option(None, "--metallic-tex",
                                                   help="Metallic map (preview surface)."),
    uv_set: Optional[str] = typer.Option(None, "--uv-set", help="UV primvar (default st)."),
    tex_scale: Optional[str] = typer.Option(None, "--tex-scale", help="Texture UV scale su,sv."),
    tex_rotate: Optional[float] = typer.Option(None, "--tex-rotate", help="Texture UV rotation (deg)."),
    tex_translate: Optional[str] = typer.Option(None, "--tex-translate", help="Texture UV offset tu,tv."),
    input_: Optional[List[str]] = typer.Option(None, "--input",
                                               help="Raw shader input name=value (repeatable)."),
    all_: bool = typer.Option(False, "--all", help="Bulk mode: select prims by rule instead of a ref."),
    type_: Optional[List[str]] = typer.Option(None, "-t", "--type",
                                              help="Bulk filter: prim type (repeatable)."),
    where: Optional[List[str]] = typer.Option(
        None, "--where", help="Bulk rule, e.g. 'name~=Conductor*' (repeatable, ANDed)."),
    under: Optional[str] = typer.Option(None, "--under", help="Bulk scope: subtree @ref/path."),
    subset: Optional[str] = typer.Option(None, "--subset",
                                         help="Bind on a named GeomSubset of the mesh (per-face)."),
    prototype: bool = typer.Option(False, "--prototype",
                                   help="Author on the instanced prim's prototype source "
                                        "(all instances pick it up)."),
    effective: bool = typer.Option(False, "--effective",
                                   help="audit: list every renderable's resolved material."),
    include_subsets: bool = typer.Option(False, "--include-subsets",
                                         help="audit: include GeomSubset coverage."),
    library: Optional[str] = typer.Option(None, "--library", help="Reference a material from a library USD."),
    library_prim: Optional[str] = typer.Option(
        None,
        "--library-prim",
        help="Exact absolute material prim path inside --library.",
    ),
    name: Optional[str] = typer.Option(None, "--name", help="Name for the new/library material."),
):
    """Set/bind material properties — on one @ref, a named --subset, or a bulk rule
    selection (--all/--type/--where/--under). Default authors a UsdPreviewSurface;
    --omnipbr / --mdl author an MDL material instead; --unbind removes the binding;
    'material audit' reports bound/unbound renderables, subsets, and unused materials."""
    if ref == "audit":
        _run("material-audit", {"effective": effective,
                                "include_subsets": include_subsets, "under": under})
        return
    mdl = mdl_module or ("OmniPBR.mdl" if omnipbr else None)
    _run("material", {"ref": ref, "color": vec(color, 3), "metallic": metallic,
                      "roughness": roughness, "opacity": opacity,
                      "clearcoat": clearcoat, "clearcoat_roughness": clearcoat_roughness,
                      "ior": ior, "emissive": vec(emissive, 3), "bind": bind,
                      "unbind": unbind, "mdl": mdl,
                      "subidentifier": subidentifier, "diffuse_texture": diffuse_texture,
                      "normal_texture": normal_texture, "orm_texture": orm_texture,
                      "roughness_texture": roughness_texture,
                      "metallic_texture": metallic_texture, "uv_set": uv_set,
                      "tex_scale": vec(tex_scale, 2), "tex_rotate": tex_rotate,
                      "tex_translate": vec(tex_translate, 2), "inputs": input_,
                      "all": all_, "type": type_, "where": where, "under": under,
                      "subset": subset, "prototype": prototype,
                      "library": library, "library_prim": library_prim, "name": name})


@app.command("material-apply")
def material_apply(
    plan: str = typer.Argument(
        ..., help="JSON material decision plan containing exact path assignments."),
    library: str = typer.Option(..., "--library", help="Material library USD."),
    path_key: str = typer.Option(
        "prim_paths",
        "--path-key",
        help="Assignment path field to apply (prim_paths/source_prim_paths/runtime_prim_paths).",
    ),
):
    """Apply every exact path/material tuple from a decision plan in one transaction.

    The plan remains the source of truth: this command does not infer targets, expand
    globs, or recalculate material choices. It accepts the material workflow decision
    patch schema and returns per-group/per-path receipts in one daemon response.
    """
    allowed_path_keys = {"prim_paths", "source_prim_paths", "runtime_prim_paths"}
    if path_key not in allowed_path_keys:
        raise ValueError(
            "--path-key must be prim_paths, source_prim_paths, or runtime_prim_paths"
        )
    plan_path = Path(plan).expanduser()
    try:
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read material decision plan {plan_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("material decision plan must be a JSON object")
    if payload.get("schema_version") != "content-agents.material-decision-patch.v1":
        raise ValueError(
            "material decision plan must use content-agents.material-decision-patch.v1"
        )
    assignments = payload.get("material_assignments")
    if not isinstance(assignments, list) or not assignments:
        raise ValueError("material decision plan requires non-empty material_assignments")
    bindings = []
    for index, assignment in enumerate(assignments):
        if not isinstance(assignment, dict):
            raise ValueError(f"material_assignments[{index}] must be an object")
        paths = assignment.get(path_key)
        if not isinstance(paths, list) or not paths:
            raise ValueError(
                f"material_assignments[{index}] requires non-empty {path_key}"
            )
        bindings.append(
            {
                "material_name": assignment.get("material_name"),
                "material_path": assignment.get("material_path"),
                "prim_paths": paths,
            }
        )
    _run("material.apply", {"library": library, "bindings": bindings})


appearance_app = typer.Typer(
    help="Clean-slate material/shader/display appearance.", no_args_is_help=True
)
app.add_typer(appearance_app, name="appearance")


@appearance_app.command("clear")
def appearance_clear():
    """Mask composed appearance in the strongest session layer.

    Blocks direct/inherited/collection/subset material bindings, direct
    surface/MDL/displacement/volume shader outputs on renderable prims, and
    displayColor/displayOpacity primvars without changing the opened source layer.
    This is one undo entry. Persist only with `save OUTPUT.usd --flatten`.
    """
    _run("appearance.clear", {})


@appearance_app.command("audit")
def appearance_audit():
    """Audit effective material, shader, display-primvar, and instance appearance."""
    _run("appearance.audit", {})


@app.command("render-probe")
def render_probe(
    required_engine: str = typer.Option(
        "ovrtx",
        "--require-engine",
        help="Required render-engine identity.",
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output-dir",
        help="Persist the tiny readiness render in this directory.",
    ),
):
    """Run a structured readiness check plus one tiny real render."""
    import json
    from pathlib import Path

    target_dir = Path(output_dir).expanduser().resolve() if output_dir else None
    if not G.server and os.environ.get(_LOCAL_GPU_FORBIDDEN_ENV) == "1":
        result = {
            "schema_version": "usd-cli.render-probe.v1",
            "required_engine": required_engine,
            "ready": False,
            "render": None,
            "error": (
                "local renderer startup is forbidden in this process; use the "
                "parent-owned usd-cli capability"
            ),
            "error_type": "parent_renderer_capability_required",
        }
        typer.echo(json.dumps(result, sort_keys=True))
        raise typer.Exit(1)

    response = dispatch(
        "render-probe",
        {
            "required_engine": required_engine,
            "output_dir": str(target_dir) if target_dir is not None else None,
        },
    )
    probe = response.data.get("probe")
    if isinstance(probe, dict):
        typer.echo(json.dumps(probe, sort_keys=True))
        raise typer.Exit(0 if probe.get("ready") is True else 1)
    raise typer.Exit(emit(response))


@app.command()
def show(refs: List[str] = typer.Argument(...)):
    """Make prim(s) visible."""
    _run("show", {"refs": refs})


@app.command()
def hide(refs: List[str] = typer.Argument(...)):
    """Hide prim(s)."""
    _run("hide", {"refs": refs})


@app.command()
def select(refs: List[str] = typer.Argument(..., help="Refs, or '-' to read from stdin.")):
    """Select prim(s). Accepts refs on stdin via '-'."""
    if refs == ["-"]:
        import sys
        refs = [line.strip() for line in sys.stdin if line.strip()]
    _run("select", {"refs": refs})


@app.command()
def deselect(refs: Optional[List[str]] = typer.Argument(None)):
    """Clear selection, or deselect specific refs."""
    _run("deselect", {"refs": refs})


@app.command()
def selection():
    """List the current selection."""
    _run("selection", {})


@app.command()
def group(refs: List[str] = typer.Argument(...), name: Optional[str] = typer.Option(None, "--name")):
    """Group prims under a new Xform."""
    _run("group", {"refs": refs, "name": name})


@app.command()
def rename(ref: str = typer.Argument(...), new_name: str = typer.Argument(...)):
    """Rename a prim."""
    _run("rename", {"ref": ref, "new_name": new_name})


@app.command("set")
def set_(ref: str = typer.Argument(...), attr: str = typer.Argument(...), value: str = typer.Argument(...)):
    """Generic property setter: attributes (value coerced) and relationships.

    If ATTR names a relationship (e.g. physics:body0 on a joint), VALUE is one or more
    comma-separated prim paths and the relationship's targets are set instead.
    """
    _run("set", {"ref": ref, "attr": attr, "value": value})


@app.command("remove-api")
def remove_api(
    ref: str = typer.Argument(..., help="Prim @ref or path."),
    api: str = typer.Argument(
        ..., help="Applied API schema, e.g. PhysicsRigidBodyAPI (short names and "
                  "SchemaAPI:instance for multiple-apply schemas accepted)."),
):
    """Remove an applied API schema and the properties it owns from a prim."""
    _run("remove-api", {"ref": ref, "api": api})


@app.command("import")
def import_(
    source: str = typer.Argument(..., help="File path or URL."),
    at: Optional[str] = typer.Option(None, "--at"),
    name: Optional[str] = typer.Option(None, "--name"),
):
    """Import an external asset."""
    # NOT absolutized: the source path is authored into the stage as a reference
    # (an absolute client path would make the saved USD machine-specific).
    _run("import", {"source": source, "at": vec(at, 3), "name": name})


@app.command()
def align(
    ref: str = typer.Argument(...),
    to: str = typer.Argument(..., metavar="REF"),
    axis: str = typer.Option("x", "--axis", help="x | y | z."),
    mode: str = typer.Option("center", "--mode", help="min | center | max."),
):
    """Align one prim to another along an axis."""
    _run("align", {"ref": ref, "to": to, "axis": axis, "mode": mode})


@app.command()
def scatter(
    ref: str = typer.Argument(...),
    on: str = typer.Option(..., "--on"),
    count: int = typer.Option(10, "--count"),
    seed: Optional[int] = typer.Option(None, "--seed"),
):
    """Scatter copies of a prim across a surface."""
    _run("scatter", {"ref": ref, "on": on, "count": count, "seed": seed})


# ── spatial queries ─────────────────────────────────────────────────────────
@app.command()
def raycast(
    origin: Optional[str] = typer.Argument(None, help="ox,oy,oz"),
    direction: Optional[str] = typer.Argument(None, help="dx,dy,dz"),
    screen: Optional[str] = typer.Option(None, "--screen", help="u,v screen-space ray."),
):
    """Cast a ray; return hit @ref, point, normal, distance."""
    _run("raycast", {"origin": vec(origin, 3, "ORIGIN"),
                     "direction": vec(direction, 3, "DIRECTION"),
                     "screen": vec(screen, 2, "--screen")})


@app.command()
def nearest(
    ref: str = typer.Argument(...),
    type_: Optional[str] = typer.Option(None, "--type"),
    count: int = typer.Option(1, "--count", "--limit"),
):
    """N nearest objects to a ref."""
    _run("nearest", {"ref": ref, "type": type_, "count": count})


@app.command()
def within(ref: str = typer.Argument(...), radius: float = typer.Argument(...)):
    """Objects within a radius of a ref."""
    _run("within", {"ref": ref, "radius": radius})


@app.command()
def overlapping(ref: str = typer.Argument(...)):
    """Objects whose bounds intersect a ref."""
    _run("overlapping", {"ref": ref})


# ── empty space ────────────────────────────────────────────────────────────────
# The verbs above find OBJECTS from their bounds; these find the ABSENCE of objects
# from actual triangles. Read-only measurements: they report regions and place
# nothing. Each region's polygon IS the usable surface — every point inside it fits
# the queried --size — so a caller samples it and acts with the verbs it already has.
space_app = typer.Typer(
    help="Find empty space: where an object fits, and where it can rest.\n\n"
         "Read-only: reports regions as a floor surface plus its bounding box, and places "
         "nothing. Every point inside a region's floor fits the size you asked "
         "about.\n"
         "  usd-cli space support --size 0.3,0.3,0.3 --container @n5\n\n"
         "Needs the 'space' extra (warp-lang)",
    no_args_is_help=True)
app.add_typer(space_app, name="space")

_SIZE_HELP = "Object size 'x,y,z' in stage units (one number = a cube)."
_CONTAINER_HELP = (
    "Measure inside this container instead of the open stage. Comma-separated, or "
    "repeat the flag. Pass one geometric container prim, or however many structural "
    "component prims it is made of — they are taken together as one thing. An assembly "
    "parent is allowed for bounds, but --exclude-others needs wall/deck components "
    "explicitly so it can remove contents safely. Bounds "
    "the search AND defines what counts as the container itself for "
    "--exclude-others. e.g. --container @l,@r,@back,@base")
_EXCLUDE_OTHERS_HELP = (
    "Drop every triangle that is not part of --container, so the answer is what the "
    "container itself affords rather than what fits around its current contents. "
    "This is how you ask for capacity. Requires --container.")
_INTERIOR_HELP = (
    "Only the space enclosed by --container, not the space above it. Off by default "
    "because an open-top container — a pallet deck, a table, a flat shelf — is loaded "
    "on its top face. Pass it for a bookshelf, where the top of the unit is reachable "
    "but nobody stacks there.")
_TAU_HELP = ("Support fraction required, 0-1. 1.0 = the whole footprint rests on "
             "surface; lower allows overhang, provided the centre of gravity stays "
             "supported and the part hanging off is clear of geometry over the "
             "object's height.")
_MARGIN_HELP = (
    "Delete support narrower than twice this, in stage units — a rim, a corner, a "
    "strip an object could only perch on. NOT an inset: a broad surface keeps its "
    "full extent, edge included. Pad --size for a stand-off from the drop."
)
_MERGE_GAP_HELP = (
    "Coincidence tolerance for the solid-layer merge, in stage units. Two surfaces in "
    "one column with less air than this between them are read as touching. The default "
    "(1 mm) is invisible unless --size is shorter than it, because a merged gap is too "
    "thin for anything taller to have used; below that, lower it or real resting "
    "surfaces are fused away."
)
_SCOPE_HELP = ("Explicit world AABB 'minx,miny,minz,maxx,maxy,maxz'. Intersected with "
               "--container when both are given.")
_CELL_HELP = "Grid resolution in stage units (default: a quarter of the object's footprint)."
_SMOOTH_HELP = ("Bridge slat/grating gaps across the object's footprint (default on). "
                "--no-smooth reports a pallet deck as unusable, which is what the raw "
                "spans say; useful for seeing what the bridging buys.")
_DEVICE_HELP = ("auto | cpu | cuda. auto (default) picks cuda whenever Warp reports a "
                "usable CUDA device — measured 4.7x on a 2.3M-triangle scene, and never "
                "slower at any size tested. Pass cpu explicitly when a result must be "
                "reproducible across machines: the two devices are each deterministic but "
                "can differ slightly on geometry dense enough to overflow the per-cell "
                "solid-layer buffer. The resolved device is echoed in the summary.")


@space_app.command("free")
def space_free(
    size: str = typer.Option(..., "--size", "-s", help=_SIZE_HELP),
    container: List[str] = typer.Option([], "--container", help=_CONTAINER_HELP),
    exclude_others: bool = typer.Option(False, "--exclude-others",
                                        help=_EXCLUDE_OTHERS_HELP),
    interior: bool = typer.Option(False, "--interior", help=_INTERIOR_HELP),
    scope: Optional[str] = typer.Option(None, "--scope", help=_SCOPE_HELP),
    cell: Optional[float] = typer.Option(None, "--cell", help=_CELL_HELP),
    stack_height: Optional[float] = typer.Option(
        None, "--stack-height",
        help="Cap each region's height, in stage units. Default: the real clearance "
             "up to whatever is overhead — 'how much is left'. Pass the object's "
             "height for a one-layer placement slab instead."),
    tau: float = typer.Option(1.0, "--tau", help=_TAU_HELP),
    margin: float = typer.Option(0.0, "--margin", help=_MARGIN_HELP),
    smooth: bool = typer.Option(True, "--smooth/--no-smooth", help=_SMOOTH_HELP),
    merge_gap: Optional[float] = typer.Option(None, "--merge-gap",
                                             help=_MERGE_GAP_HELP),
    device: str = typer.Option("auto", "--device", help=_DEVICE_HELP),
):
    """How much room is left, ranked by volume.

    A container's capacity is this verb with --container and --exclude-others:

        usd-cli space free --size 0.3,0.3,0.3 --exclude-others --container @cab_l,@cab_r,@cab_back,@cab_base
    """
    _run("space.free", {"size": size, "container": list(container) or None,
                        "exclude_others": exclude_others, "interior": interior,
                        "scope": scope, "cell": cell, "stack_height": stack_height,
                        "tau": tau, "margin": margin,
                        "smooth": smooth, "merge_gap": merge_gap, "device": device},
         preserve_false=("smooth",))


@space_app.command("support")
def space_support(
    size: str = typer.Option(..., "--size", "-s", help=_SIZE_HELP),
    container: List[str] = typer.Option([], "--container", help=_CONTAINER_HELP),
    exclude_others: bool = typer.Option(False, "--exclude-others",
                                        help=_EXCLUDE_OTHERS_HELP),
    interior: bool = typer.Option(False, "--interior", help=_INTERIOR_HELP),
    scope: Optional[str] = typer.Option(None, "--scope", help=_SCOPE_HELP),
    cell: Optional[float] = typer.Option(None, "--cell", help=_CELL_HELP),
    tau: float = typer.Option(1.0, "--tau", help=_TAU_HELP),
    margin: float = typer.Option(0.0, "--margin", help=_MARGIN_HELP),
    smooth: bool = typer.Option(True, "--smooth/--no-smooth", help=_SMOOTH_HELP),
    merge_gap: Optional[float] = typer.Option(None, "--merge-gap",
                                             help=_MERGE_GAP_HELP),
    device: str = typer.Option("auto", "--device", help=_DEVICE_HELP),
):
    """Where one object of --size can rest stably, ranked by support area."""
    _run("space.support", {"size": size, "container": list(container) or None,
                           "exclude_others": exclude_others, "interior": interior,
                           "scope": scope, "cell": cell, "tau": tau, "margin": margin,
                           "smooth": smooth, "merge_gap": merge_gap,
                           "device": device}, preserve_false=("smooth",))


@app.command()
def distance(a: str = typer.Argument(...), b: str = typer.Argument(...)):
    """Distance between two objects."""
    _run("distance", {"a": a, "b": b})


# directional queries — registered in a loop to avoid six identical bodies
def _make_directional(direction: str):
    def _cmd(
        ref: str = typer.Argument(...),
        type_: Optional[str] = typer.Option(None, "--type"),
        count: Optional[int] = typer.Option(None, "--count"),
    ):
        _run(direction, {"ref": ref, "type": type_, "count": count})
    _cmd.__doc__ = f"Objects positioned {direction} relative to a ref."
    return _cmd


for _dir in ("above", "below", "left", "right", "front", "behind"):
    app.command(name=_dir)(_make_directional(_dir))


# ── camera (noun-verb) ──────────────────────────────────────────────────────
camera = typer.Typer(help="Camera control.", no_args_is_help=True)
app.add_typer(camera, name="camera")


@camera.command("list")
def camera_list():
    """List available cameras."""
    _run("camera.list", {})


@camera.command("use")
def camera_use(ref: str = typer.Argument(...)):
    """Switch active camera."""
    _run("camera.use", {"ref": ref})


@camera.command("look-at")
def camera_look_at(ref: str = typer.Argument(...)):
    """Point camera at an object."""
    _run("camera.look-at", {"ref": ref})


@camera.command("orbit")
def camera_orbit(
    ref: str = typer.Argument(...),
    az: Optional[float] = typer.Option(None, "--az", "--azimuth"),
    el: Optional[float] = typer.Option(None, "--el", "--elevation"),
    dist: Optional[float] = typer.Option(None, "--dist", "--distance"),
):
    """Orbit around an object."""
    _run("camera.orbit", {"ref": ref, "az": az, "el": el, "dist": dist})


@camera.command("fit")
def camera_fit(refs: List[str] = typer.Argument(...)):
    """Auto-frame objects."""
    _run("camera.fit", {"refs": refs})


@camera.command("pan")
def camera_pan(delta: str = typer.Argument(..., help="dx,dy")):
    """Pan in screen space."""
    _run("camera.pan", {"delta": vec(delta, 2)})


@camera.command("zoom")
def camera_zoom(factor: float = typer.Argument(...)):
    """Zoom in/out."""
    _run("camera.zoom", {"factor": factor})


@camera.command("create")
def camera_create(
    name: Optional[str] = typer.Option(None, "--name"),
    at: Optional[str] = typer.Option(None, "--at", help="Position: 'x,y,z' or a prim ref/path."),
    look_at: Optional[str] = typer.Option(None, "--look-at",
                                          help="Aim point: 'x,y,z' or a prim ref/path to frame."),
    focal: Optional[float] = typer.Option(None, "--focal", help="Focal length in mm."),
    aperture: Optional[float] = typer.Option(
        None,
        "--aperture",
        help="Horizontal and vertical aperture in mm.",
    ),
):
    """Create a new camera. --at/--look-at take coordinates OR a prim ref/path."""
    # Pass raw strings: the session resolves a "x,y,z" literal or a prim ref/path to a
    # world point, so `--look-at @n1` works instead of crashing the float parser.
    _run(
        "camera.create",
        {
            "name": name,
            "at": at,
            "look_at": look_at,
            "focal": focal,
            "aperture": aperture,
        },
    )


@camera.command("coverage")
def camera_coverage(
    scope: Optional[str] = typer.Option(None, "--scope", help="Floor/region @ref or path."),
    camera_ref: Optional[List[str]] = typer.Option(
        None,
        "--camera",
        help="Camera or rig-subtree @ref/path; repeat for multiple selections.",
    ),
    cameras: Optional[str] = typer.Option(
        None, "--cameras", help="Comma-separated camera @refs/paths."
    ),
    target: float = typer.Option(0.95, "--target", help="Required coverage fraction."),
    per_cell: int = typer.Option(1, "--per-cell", help="Required views per sampled cell."),
    grid: int = typer.Option(32, "--grid", help="Cells along the longer scope axis."),
    cell_size: Optional[float] = typer.Option(
        None, "--cell-size", help="Grid cell size in canonical metres."
    ),
    device: str = typer.Option("cpu", "--device", help="Warp device, e.g. cpu or cuda:0."),
    output: Optional[str] = typer.Option(None, "-o", "--output", help="Write full JSON report."),
    detach: bool = typer.Option(
        False,
        "--detach",
        help="Return a job id; collect with `usd-cli wait` or stop with `usd-cli cancel JOB`.",
    ),
):
    """Measure existing-camera coverage without changing the stage."""

    selected = list(camera_ref or [])
    if cameras:
        selected.extend(item.strip() for item in cameras.split(",") if item.strip())
    _run(
        "camera.coverage",
        {
            "scope": scope,
            "cameras": selected or None,
            "target": target,
            "per_cell": per_cell,
            "grid": grid,
            "cell_size": cell_size,
            "device": device,
            "output": _abspath(output),
            "detach": detach,
        },
    )


@camera.command("place")
def camera_place(
    method: str = typer.Option(..., "--method", help="max_coverage or look_at."),
    scope: Optional[str] = typer.Option(None, "--scope", help="Region for max_coverage."),
    target: Optional[str] = typer.Option(None, "--target", help="Object for look_at."),
    target_coverage: Optional[float] = typer.Option(
        None, "--target-coverage", help="Coverage target (default: 0.95)."
    ),
    per_cell: Optional[int] = typer.Option(
        None, "--per-cell", help="Views required per grid cell (default: 1)."
    ),
    max_cameras: Optional[int] = typer.Option(
        None, "--max-cameras", help="Coverage camera limit (default: 8)."
    ),
    cameras: Optional[int] = typer.Option(
        None, "--cameras", help="Exact look_at camera count (default: 4)."
    ),
    grid: Optional[int] = typer.Option(
        None, "--grid", help="Coverage grid long-axis cells (default: 32)."
    ),
    patch_size: Optional[float] = typer.Option(
        None,
        "--patch-size",
        "--cell-size",
        help="Coverage patch size in canonical metres; mutually exclusive with --grid.",
    ),
    candidates: int = typer.Option(64, "--candidates"),
    height: Optional[float] = typer.Option(
        None,
        "--height",
        help="Metres above floor (max_coverage default: 3) or target-center "
        "offset (look_at default: 0).",
    ),
    min_height: Optional[float] = typer.Option(
        None,
        "--min-height",
        help="Minimum look_at height above the target center in canonical metres.",
    ),
    max_height: Optional[float] = typer.Option(
        None,
        "--max-height",
        help="Maximum look_at height above the target center in canonical metres.",
    ),
    standoff: Optional[float] = typer.Option(
        None,
        "--standoff",
        help="Coverage perimeter standoff in metres (default: 0.5).",
    ),
    minimum_gain: Optional[float] = typer.Option(
        None, "--minimum-gain", help="Minimum coverage marginal gain (default: 0)."
    ),
    min_look_down: Optional[float] = typer.Option(
        None, "--min-look-down", help="Minimum pitch below horizontal in degrees."
    ),
    max_look_down: Optional[float] = typer.Option(
        None, "--max-look-down", help="Maximum pitch below horizontal in degrees."
    ),
    yaw_ranges: Optional[str] = typer.Option(
        None, "--yaw-ranges", help="Allowed look_at bearings, e.g. '0,180;270,360'."
    ),
    occlusion_threshold: Optional[float] = typer.Option(
        None, "--occlusion-threshold", help="Maximum look_at occlusion (default: 0.4)."
    ),
    min_distance: Optional[float] = typer.Option(None, "--min-distance"),
    max_distance: Optional[float] = typer.Option(None, "--max-distance"),
    min_x: Optional[float] = typer.Option(
        None, "--min-x", help="Minimum look_at camera X in canonical metres."
    ),
    max_x: Optional[float] = typer.Option(
        None, "--max-x", help="Maximum look_at camera X in canonical metres."
    ),
    min_y: Optional[float] = typer.Option(
        None, "--min-y", help="Minimum look_at camera Y in canonical metres."
    ),
    max_y: Optional[float] = typer.Option(
        None, "--max-y", help="Maximum look_at camera Y in canonical metres."
    ),
    allow_fewer: bool = typer.Option(False, "--allow-fewer"),
    seed: int = typer.Option(0, "--seed"),
    focal: Optional[float] = typer.Option(None, "--focal"),
    aperture: float = typer.Option(36.0, "--aperture"),
    device: str = typer.Option("cpu", "--device"),
    preview: bool = typer.Option(
        False, "--preview", help="Explicitly request no stage mutation."
    ),
    author_under: Optional[str] = typer.Option(
        None, "--author-under", help="Atomically author the accepted rig below this path."
    ),
    on_existing: str = typer.Option(
        "error", "--on-existing", help="error, replace, or append."
    ),
    output: Optional[str] = typer.Option(
        None, "-o", "--output", help="Write full JSON report."
    ),
    detach: bool = typer.Option(
        False,
        "--detach",
        help="Return a job id; collect with `usd-cli wait` or stop with `usd-cli cancel JOB`.",
    ),
):
    """Preview or author a coverage-aware multi-camera rig."""

    normalized_method = method.strip().lower()
    _run(
        "camera.place",
        {
            "method": normalized_method,
            "scope": scope,
            "target": target,
            "target_coverage": target_coverage,
            "per_cell": per_cell,
            "max_cameras": max_cameras,
            "cameras": cameras,
            "grid": grid,
            "patch_size": patch_size,
            "candidates": candidates,
            "height": height,
            "min_height": min_height,
            "max_height": max_height,
            "standoff": standoff,
            "minimum_gain": minimum_gain,
            "min_look_down": min_look_down,
            "max_look_down": max_look_down,
            "yaw_ranges": yaw_ranges,
            "occlusion_threshold": occlusion_threshold,
            "min_distance": min_distance,
            "max_distance": max_distance,
            "min_x": min_x,
            "max_x": max_x,
            "min_y": min_y,
            "max_y": max_y,
            "allow_fewer": allow_fewer,
            "seed": seed,
            "focal": focal,
            "aperture": aperture,
            "device": device,
            "preview": preview,
            "author_under": author_under,
            "on_existing": on_existing,
            "output": _abspath(output),
            "detach": detach,
        },
    )


@camera.command("rig-export")
def camera_rig_export(
    rig: str = typer.Argument(..., help="Camera rig @ref/path."),
    res: str = typer.Option("1920x1080", "--res"),
    include_visibility: bool = typer.Option(False, "--include-visibility"),
    verify: bool = typer.Option(False, "--verify", help="Require digest-bound OVRTX evidence."),
    scope: Optional[str] = typer.Option(
        None, "--scope", help="Coverage scope if not stored on rig."
    ),
    grid: int = typer.Option(32, "--grid"),
    device: str = typer.Option("cpu", "--device"),
    output: str = typer.Option("rig.json", "-o", "--output"),
    detach: bool = typer.Option(
        False,
        "--detach",
        help="Return a job id; collect with `usd-cli wait` or stop with `usd-cli cancel JOB`.",
    ),
):
    """Export canonical intrinsics, extrinsics, visibility, and verification evidence."""

    _run(
        "camera.rig-export",
        {
            "rig": rig,
            "res": resolution(res),
            "include_visibility": include_visibility,
            "verify": verify,
            "scope": scope,
            "grid": grid,
            "device": device,
            "output": _abspath(output),
            "detach": detach,
        },
    )


# ── history (noun-verb for checkpoint) ──────────────────────────────────────
@app.command()
def undo(n: int = typer.Argument(1)):
    """Undo the last N operations."""
    _run("undo", {"n": n})


@app.command()
def redo(n: int = typer.Argument(1)):
    """Redo the last N operations."""
    _run("redo", {"n": n})


@app.command()
def history(n: Optional[int] = typer.Argument(None)):
    """Show the last N operations."""
    _run("history", {"n": n})


checkpoint = typer.Typer(help="Named scene states.", no_args_is_help=True)
app.add_typer(checkpoint, name="checkpoint")


@checkpoint.command("save")
@checkpoint.command("create", hidden=True)  # habitual alias agents reach for
def checkpoint_save(
    name: str = typer.Argument(...),
    full: bool = typer.Option(False, "--full", help="Flatten the whole stage (default: edit-layer delta)."),
):
    """Save a named checkpoint as a USD layer under .usd-cli/checkpoints/."""
    _run("checkpoint.save", {"name": name, "full": full})


@checkpoint.command("load")
def checkpoint_load(
    name: str = typer.Argument(...),
    detach: bool = typer.Option(
        False, "--detach",
        help="Return a job id immediately (big-scene restores recompose for "
             "minutes); collect with `usd-cli wait`."),
):
    """Restore a named checkpoint."""
    _run("checkpoint.load", {"name": name, "detach": detach})


@checkpoint.command("list")
def checkpoint_list():
    """List checkpoints."""
    _run("checkpoint.list", {})


@checkpoint.command("delete")
def checkpoint_delete(name: str = typer.Argument(...)):
    """Delete a checkpoint."""
    _run("checkpoint.delete", {"name": name})


# ── scene I/O ───────────────────────────────────────────────────────────────
def _abspath(path: Optional[str]) -> Optional[str]:
    """Resolve a user path against the CLIENT's cwd before it crosses to the daemon.

    The daemon may run with a different working directory (it lives per project,
    not per shell), so a relative path sent verbatim used to resolve against the
    daemon's cwd — opening or writing somewhere the user didn't mean. Symlinks are
    NOT resolved (abspath, not realpath): the daemon's guards key on realpath and
    must see the path the user named. Resolver URIs (omniverse://, usdz-internal
    `pkg.usdz[...]`) pass through untouched — they are not filesystem paths."""
    if not path or "://" in path or "[" in path:
        return path
    import os

    return os.path.abspath(os.path.expanduser(path))


@app.command()
def open(
    file: str = typer.Argument(..., help="Path to .usd/.usda/.usdc/.usdz."),
    read_only: bool = typer.Option(
        False, "--read-only",
        help="Open for parallel inspection: bypasses the one-writer-per-file guard "
             "and blocks every mutating command in this session. Readers share the "
             "writer's LIVE layer, so they see its unsaved edits as they happen; "
             "any number of readers may coexist with one writer."),
    force_reload: bool = typer.Option(
        False, "--force-reload",
        help="Re-open the file this session already holds even if it has unsaved "
             "edits, deliberately DISCARDING them and reloading from disk."),
):
    """Load a USD file."""
    _run("open", drop_none({"file": _abspath(file), "read_only": read_only,
                            "force_reload": force_reload}))


@app.command()
def save(
    path: Optional[str] = typer.Argument(None),
    flatten: bool = typer.Option(
        False, "--flatten",
        help="Composition-flatten the stage into one layer: localizes composition "
             "arcs — sublayers, references (e.g. `material --library`), payloads — "
             "into the saved file. External texture/asset FILES remain referenced, "
             "not embedded; save to .usdz to package them."),
    detach: bool = typer.Option(
        False, "--detach",
        help="Return a job id immediately (big-scene saves recompose for "
             "minutes); collect with `usd-cli wait`."),
):
    """Save the current stage (atomic: temp sibling, verify read-back, then rename;
    a symlink destination is written through to its target, preserving the link)."""
    _run("save", {"path": _abspath(path), "flatten": flatten, "detach": detach})


@app.command("viewer-snapshot", hidden=True)
def viewer_snapshot(
    path: str = typer.Argument(..., help="Viewer-owned flattened snapshot path."),
    since_revision: int = typer.Option(
        -1,
        "--since-revision",
        min=-1,
        help="Skip publication when the live stage has not changed since this revision.",
    ),
    line_geometry: Optional[str] = typer.Option(
        None,
        "--line-geometry",
        help="Viewer-owned NPZ path for the realtime technical-line pass.",
    ),
):
    """Publish an atomic, flattened observation copy of the live session stage."""

    _run(
        "viewer.snapshot",
        {
            "path": _abspath(path),
            "since_revision": since_revision,
            "line_geometry": _abspath(line_geometry),
        },
    )


@app.command()
def export(format: str = typer.Argument(..., help="obj | glb | fbx."), path: Optional[str] = typer.Argument(None)):
    """Export to another format."""
    _run("export", {"format": format, "path": _abspath(path)})


@app.command()
def sublayers(
    drop_dead: bool = typer.Option(False, "--drop-dead", help="Remove sublayers that don't resolve (dead omniverse:// / missing-file arcs)."),
):
    """List the root layer's sublayers; --drop-dead strips unresolvable ones."""
    _run("sublayers", {"drop_dead": drop_dead})


@app.command()
def new(
    up: Optional[str] = typer.Option(None, "--up", help="Up axis: Y (default) or Z. Use Z when referencing Z-up source assets."),
    meters_per_unit: Optional[float] = typer.Option(None, "--meters-per-unit", help="Stage metersPerUnit (e.g. 0.01 for cm)."),
):
    """Create an empty scene with default lighting + camera."""
    _run("new", {"up": up, "meters_per_unit": meters_per_unit})


@app.command()
def info():
    """Scene metadata: up-axis, units, frame range, prim count, layers."""
    _run("info", {})


# ── lifecycle (noun-verb for server) ────────────────────────────────────────
server_app = typer.Typer(
    help=(
        "Daemon lifecycle. Externally owned sessions reject child "
        "start/stop/restart operations."
    ),
    no_args_is_help=True,
)
app.add_typer(server_app, name="server")


# Daemon lifecycle is a client-side concern (managing the process), so these bypass the
# command dispatch and talk to usd_cli.daemon directly.
def _daemon_op(fn_name: str) -> None:
    from usd_cli import daemon
    from usd_core import load_config
    try:
        print(getattr(daemon, fn_name)(load_config()))
    except daemon.DaemonStopRefused as exc:
        # A refusal is a decision, not a crash: the daemon lifecycle deliberately
        # declines to signal a process it cannot authenticate as ours. Report it as
        # one line with exit 1 (the operation failed) rather than a traceback.
        print(f"error: {exc}", file=sys.stderr)
        raise typer.Exit(EXIT_RUNTIME) from exc


def _require_child_lifecycle_authority(action: str) -> None:
    from usd_core import load_config

    config = load_config()
    externally_owned = (
        os.environ.get(_EXTERNAL_LIFECYCLE_ENV)
        or config.server.get("lifecycle_owner") == "external"
    )
    bootstrap = (
        action == "start"
        and os.environ.get(_EXTERNAL_LIFECYCLE_BOOTSTRAP_ENV) == "1"
        and not config.server_state_path.exists()
    )
    if externally_owned and not bootstrap:
        print(
            "error: externally owned usd-cli sessions forbid child "
            f"`server {action}`",
            file=sys.stderr,
        )
        raise typer.Exit(EXIT_RUNTIME)


@server_app.command("status")
def server_status():
    """Show daemon status."""
    _daemon_op("status")


@server_app.command("start")
def server_start():
    """Start the daemon for this project."""
    _require_child_lifecycle_authority("start")
    _daemon_op("start")


@server_app.command("stop")
def server_stop():
    """Stop the daemon."""
    _require_child_lifecycle_authority("stop")
    _daemon_op("stop")


@server_app.command("restart")
def server_restart():
    """Restart the daemon."""
    _require_child_lifecycle_authority("restart")
    from usd_cli import daemon
    from usd_core import load_config
    cfg = load_config()
    try:
        print(daemon.stop(cfg))
    except daemon.DaemonStopRefused as exc:
        # Never start on a refused stop: the old daemon may still be live and
        # serving, and a second one would orphan it and hijack the state file.
        print(f"error: {exc}", file=sys.stderr)
        print("note: not starting a replacement while the existing daemon is "
              "unverified — `usd-cli server status` to inspect it", file=sys.stderr)
        raise typer.Exit(EXIT_RUNTIME) from exc
    print(daemon.start(cfg))


@server_app.command("release-session")
def server_release_session(
    name: Optional[str] = typer.Option(
        None,
        "--name",
        help="Exact named session to release (required; ambient session state is ignored).",
    ),
):
    """Release one explicitly named session without stopping the daemon."""

    if name is None or not name.strip():
        raise typer.BadParameter("an explicit session name is required", param_hint="--name")
    _run("server.release-session", {"name": name})


# ── remote render backend (client-side; bypasses dispatch) ─────────────────────────
remote_app = typer.Typer(help="Run renders on a remote OVRTX GPU service.", no_args_is_help=True)
app.add_typer(remote_app, name="remote")


@remote_app.command("serve-cmd")
def remote_serve_cmd(
    bare: bool = typer.Option(False, "--bare", help="Bare-metal (uvicorn) instead of Docker."),
    port: int = typer.Option(8000, "--port", help="Port the service should listen on."),
):
    """Print commands to run the managed OVRTX adapter on a remote GPU host."""
    from usd_cli import remote
    typer.echo(remote.serve_cmd(bare=bare, port=port))


def _remote_config_target(use_global: bool):
    """The config.toml `remote configure` should write (project default, else global)."""
    from usd_cli import remote
    from usd_core import load_config

    cfg = load_config()
    if use_global:
        return remote._global_config_path()
    return remote._project_config_path(cfg) or remote._global_config_path()


def _write_remote_primary(url: str, api_key: Optional[str], use_global: bool):
    """Persist renderer=remote + the primary endpoint, preserving every other key.

    With a [[render.backends]] pool present, pool resolution IGNORES the legacy
    remote_url/remote_api_key fields, so the primary is ALSO written as pool entry
    zero (moved there if the URL is already pooled, inserted otherwise) — a plain
    `remote configure URL` used to update only the legacy fields, probing the new
    server while renders kept going to the old pool. Credentials are preserved only
    for the same endpoint: re-pointing at a new URL without an environment-supplied
    key never carries the old backend's key over to the new server."""
    from usd_core.config import (_read_toml, config_file_lock, dump_toml,
                                 write_config_atomic)

    path = _remote_config_target(use_global)
    url = url.rstrip("/")
    with config_file_lock(path):
        existing = _read_toml(path)
        render = dict(existing.get("render", {}))
        backends = [dict(b) for b in render.get("backends", []) if isinstance(b, dict)]
        stored = next((str(b.get("api_key") or "") for b in backends
                       if str(b.get("url", "")).rstrip("/") == url), "")
        if not stored and str(render.get("remote_url", "") or "").rstrip("/") == url:
            stored = str(render.get("remote_api_key", "") or "")
        key = api_key or stored
        render["renderer"] = "remote"
        render["remote_url"] = url
        if key:
            render["remote_api_key"] = key
        else:
            render.pop("remote_api_key", None)
        if backends:  # keep the resolved pool in lockstep with the legacy fields
            entry: dict = {"url": url}
            if key:
                entry["api_key"] = key
            render["backends"] = [entry] + [
                b for b in backends if str(b.get("url", "")).rstrip("/") != url]
        existing["render"] = render
        write_config_atomic(path, dump_toml(existing))
    return path


def _append_render_backend(url: str, api_key: Optional[str], use_global: bool):
    """`remote configure --add`: append a [[render.backends]] entry, creating the array
    (seeded with the existing primary) when absent. Re-adding a known URL updates it
    in place, PRESERVING its stored api_key unless one is supplied through the
    environment. Also seeds
    renderer/remote_url when this is the first backend ever configured, so the
    legacy single-backend fields stay meaningful."""
    from usd_core.config import (_read_toml, config_file_lock, dump_toml,
                                 write_config_atomic)

    path = _remote_config_target(use_global)
    url = url.rstrip("/")
    with config_file_lock(path):
        existing = _read_toml(path)
        render = dict(existing.get("render", {}))
        backends = [dict(b) for b in render.get("backends", []) if isinstance(b, dict)]
        if not backends and render.get("remote_url"):
            primary = {"url": str(render["remote_url"]).rstrip("/")}
            if render.get("remote_api_key"):
                primary["api_key"] = render["remote_api_key"]
            backends.append(primary)
        for b in backends:
            if str(b.get("url", "")).rstrip("/") == url:
                b["url"] = url
                if api_key:  # only an explicit key replaces the stored one
                    b["api_key"] = api_key
                break
        else:
            entry: dict = {"url": url}
            if api_key:
                entry["api_key"] = api_key
            backends.append(entry)
        render["backends"] = backends
        render["renderer"] = "remote"
        if not render.get("remote_url"):  # first backend ever → it is also the primary
            render["remote_url"] = url
            if api_key:
                render["remote_api_key"] = api_key
        existing["render"] = render
        write_config_atomic(path, dump_toml(existing))
    return path, len(backends)


@remote_app.command("configure")
def remote_configure(
    url: str = typer.Argument(..., help="Remote service URL, e.g. http://gpu-host:8000."),
    add: bool = typer.Option(False, "--add",
                             help="Append to the configured backend pool instead of "
                                  "replacing the primary (parallel rendering)."),
    global_: bool = typer.Option(False, "--global", help="Write to ~/.config/usd-cli (default: project)."),
    no_test: bool = typer.Option(False, "--no-test", help="Skip the /health probe."),
    timeout: float = typer.Option(10.0, "--timeout", help="Health-probe timeout (seconds)."),
):
    """Set renderer=remote + remote_url in config, then probe the service's /health.

    Read a bearer key from OVRTX_API_KEY (or USD_CLI_RENDER_REMOTE_API_KEY), never
    from a command-line argument.
    """
    from usd_cli import remote

    api_key = os.environ.get("OVRTX_API_KEY") or os.environ.get(
        "USD_CLI_RENDER_REMOTE_API_KEY"
    )
    if add:
        path, n = _append_render_backend(url, api_key, use_global=global_)
        typer.echo(f"added render backend {url.rstrip('/')} "
                   f"({n} in the [[render.backends]] pool) → {path}")
    else:
        path = _write_remote_primary(url, api_key, use_global=global_)
        typer.echo(f"wrote renderer=remote, remote_url={url.rstrip('/')} → {path}")

    if no_test:
        raise typer.Exit(0)

    health = remote.probe(url, api_key=api_key, timeout=timeout)
    if health.get("error") is not None:
        typer.echo(f"⚠ could not reach {url}: {health['error']}", err=True)
        typer.echo("  (config was still written — start the service, then `usd-cli remote configure` again to verify)")
        raise typer.Exit(1)

    ready = health.get("gpu_initialized")
    typer.echo(f"✓ reachable — status={health.get('status')}, "
               f"gpu_initialized={ready}, daemon_running={health.get('daemon_running')}")

    from usd_core.remote_protocol import PROTOCOL_VERSION
    backend_pv = health.get("protocol_version")
    if backend_pv == PROTOCOL_VERSION:
        typer.echo(f"✓ protocol v{PROTOCOL_VERSION} — backend matches this client")
    else:
        got = f"v{backend_pv}" if backend_pv else "unversioned (predates version checking)"
        typer.echo(f"⚠ protocol mismatch: backend is {got}, this client requires "
                   f"v{PROTOCOL_VERSION} — remote render/physics will refuse to run. "
                   "Re-deploy the service from this checkout (`usd-cli remote serve-cmd`).",
                   err=True)
    if not ready:
        typer.echo("  GPU still warming up (cold start can take minutes); renders will 503 until ready.")
    typer.echo("  note: one service serves many sessions but renders them serially "
               "(single GPU daemon); run one service per GPU for parallelism — add "
               "each with `usd-cli remote configure URL --add` to shard renders across them.")
    raise typer.Exit(0)


@remote_app.command("backends")
def remote_backends(
    timeout: float = typer.Option(5.0, "--timeout", help="Per-backend probe timeout (seconds)."),
    deep: bool = typer.Option(
        False, "--deep",
        help="Also run a tiny REAL render per backend (a cube, 64x64) — proves GPU "
             "render capability, not just HTTP reachability. Round 7: /live stayed "
             "green while every render 500'd (the containers had lost GPU access)."),
):
    """List the render backend pool with a /live probe per backend."""
    import httpx

    from usd_core import load_config
    from usd_core.config import resolve_render_backends
    from usd_core.remote_protocol import PROTOCOL_VERSION

    cfg = load_config()
    pool = resolve_render_backends(cfg.render)
    if not pool:
        typer.echo("no remote render backends configured — add one with "
                   "`OVRTX_API_KEY=K usd-cli remote configure URL [--add]` "
                   "(credentials are accepted only through environment injection)",
                   err=True)
        raise typer.Exit(1)

    def _deep_probe(url: str, api_key: str | None) -> str:
        """'ok (N.Ns)' or the failure reason — one minimal real render."""
        import time as _time
        cube = (
            '#usda 1.0\n'
            '(\n    defaultPrim = "World"\n)\n'
            'def Xform "World"\n{\n'
            '    def Cube "probe"\n    {\n        double size = 1\n    }\n'
            '    def Camera "cam"\n    {\n'
            '        matrix4d xformOp:transform = ( (0.707, -0.408, 0.577, 0), '
            '(0, 0.816, 0.577, 0), (-0.707, -0.408, 0.577, 0), (3, 3, 3, 1) )\n'
            '        uniform token[] xformOpOrder = ["xformOp:transform"]\n'
            '    }\n}\n')
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        t0 = _time.perf_counter()
        try:
            resp = httpx.post(f"{url}/render",
                              json={"usd": cube, "cameras": ["/World/cam"],
                                    "image_width": 64, "image_height": 64,
                                    "mode": "fast"},
                              headers=headers, timeout=max(timeout, 120.0))
            if resp.status_code == 200 and resp.json().get("results"):
                return f"render ok ({_time.perf_counter() - t0:.1f}s)"
            detail = ""
            try:
                detail = str(resp.json().get("detail", ""))[:120]
            except Exception:  # noqa: BLE001
                detail = resp.text[:120]
            return f"render FAILED (HTTP {resp.status_code}: {detail})"
        except Exception as exc:  # noqa: BLE001 — probe result, not a crash
            return f"render FAILED ({type(exc).__name__}: {exc})"

    usable = 0
    for backend in pool:
        url = backend["url"]
        try:
            resp = httpx.get(f"{url}/live", timeout=timeout)
            pv = resp.json().get("protocol_version") if resp.status_code == 200 else None
        except Exception as exc:  # noqa: BLE001 — probe result, not a crash
            typer.echo(f"✗ {url} — unreachable ({type(exc).__name__}: {exc})")
            continue
        if pv == PROTOCOL_VERSION:
            extra = ""
            if deep:
                verdict = _deep_probe(url, backend.get("api_key") or None)
                extra = f" · {verdict}"
                if "FAILED" in verdict:
                    typer.echo(f"⚠ {url} — reachable, protocol v{pv}{extra}")
                    continue
            usable += 1
            typer.echo(f"✓ {url} — reachable, protocol v{pv} (matches this client){extra}")
        else:
            got = f"v{pv}" if pv is not None else "unversioned (predates version checking)"
            typer.echo(f"⚠ {url} — reachable, protocol {got}; this client requires "
                       f"v{PROTOCOL_VERSION} — it will be skipped at render time")
    typer.echo(f"{usable} of {len(pool)} backend(s) usable "
               f"(pool source: {'[[render.backends]]' if cfg.render.get('backends') else 'remote_url (single)'})")
    raise typer.Exit(0 if usable else 1)


# ── live viewport, human-in-the-loop, watch ──────────────────────────────────────
# NOT YET IMPLEMENTED — hidden from the CLI until the Session methods exist. These
# dispatch to session.connect/disconnect/attach/handoff/resume/watch_* which are not
# implemented, so exposing them would only surface "unknown command" errors. Re-enable
# each command as its Session method lands.
#
# @app.command()
# def connect():
#     """Switch to a headed live viewport."""
#     _run("connect", {})
#
#
# @app.command()
# def disconnect():
#     """Return to headless mode."""
#     _run("disconnect", {})
#
#
# @app.command()
# def attach(endpoint: str = typer.Argument(...)):
#     """Attach to an already-running session."""
#     _run("attach", {"endpoint": endpoint})
#
#
# # ── human-in-the-loop ──────────────────────────────────────────────────────
# @app.command()
# def handoff(message: Optional[str] = typer.Argument(None)):
#     """Open the viewport for manual editing; agent pauses."""
#     _run("handoff", {"message": message})
#
#
# @app.command()
# def resume():
#     """Re-snapshot after a handoff and continue with fresh refs."""
#     _run("resume", {})
#
#
# watch = typer.Typer(help="Passively observe human editing.", no_args_is_help=True)
# app.add_typer(watch, name="watch")
#
#
# @watch.command("start")
# def watch_start():
#     """Begin observing; read-only, blocks agent writes."""
#     _run("watch.start", {})
#
#
# @watch.command("stop")
# def watch_stop():
#     """End observation; return the accumulated change log."""
#     _run("watch.stop", {})
#
#
# @watch.command("status")
# def watch_status():
#     """Show watch status."""
#     _run("watch.status", {})


# ── query & search ─────────────────────────────────────────────────────────
@app.command()
def find(
    name: Optional[str] = typer.Option(None, "--name"),
    type_: Optional[str] = typer.Option(None, "--type"),
    material: Optional[str] = typer.Option(None, "--material"),
    has_attr: Optional[str] = typer.Option(None, "--has-attr"),
    in_box: Optional[List[str]] = typer.Option(None, "--in-box", help="x1,y1,z1 x2,y2,z2 (two values)."),
    in_sphere: Optional[List[str]] = typer.Option(None, "--in-sphere", help="cx,cy,cz then radius."),
    visible: bool = typer.Option(False, "--visible"),
    hidden: bool = typer.Option(False, "--hidden"),
    selected: bool = typer.Option(False, "--selected"),
    under: Optional[str] = typer.Option(None, "--under", help="Scope to a subtree (@ref/path)."),
    where: Optional[List[str]] = typer.Option(
        None, "--where",
        help="Rule expression(s), same syntax as `material --where` "
             "(name~=Glob*, attr:NAME==VALUE, ...; repeatable, ANDed)."),
    count: bool = typer.Option(False, "--count", help="Counts by type only — no listing."),
):
    """Search the scene. With -q, prints bare refs for piping."""
    _run("find", {
        "name": name, "type": type_, "material": material, "has_attr": has_attr,
        "in_box": in_box, "in_sphere": in_sphere,
        "visible": visible, "hidden": hidden, "selected": selected,
        "under": under, "where": where, "count_only": count,
    })


@app.command()
def wait(
    job: Optional[str] = typer.Argument(
        None, help="Job id from a --detach command (e.g. j1). Default: the newest job."),
    timeout: float = typer.Option(1500.0, "--timeout", help="Max seconds to block."),
):
    """Block until a detached job finishes; prints its full result."""
    _run("wait", {"job": job or "", "timeout": timeout})


@app.command()
def jobs():
    """List detached jobs (running + finished)."""
    _run("jobs", {})


@app.command()
def cancel(job: str = typer.Argument(..., help="Detached camera-analysis job id.")):
    """Cooperatively cancel a detached camera-analysis job."""
    _run("cancel", {"job": job})


@app.command()
def verify(
    file: Optional[str] = typer.Argument(None, help="USD to verify (default: the open stage's file)."),
    strict: bool = typer.Option(
        False, "--strict",
        help="Warnings FAIL the verdict: external deps (not self-contained) "
             "and hidden renderables demote PASS to FAIL."),
):
    """One-shot deliverable check: fresh open + sublayers + validate + material audit."""
    _run("verify", {"file": _abspath(file), "strict": strict})


@app.command()
def stats():
    """Scene overview: prim counts by type (instance-aware), meshes, instances, bounds."""
    _run("stats", {})


@app.command()
def describe(ref: Optional[str] = typer.Argument(None)):
    """Natural-language description of the scene or an object."""
    _run("describe", {"ref": ref})


@app.command()
def properties(
    ref: str = typer.Argument(..., help="@ref or /path to introspect."),
    attr: Optional[list[str]] = typer.Argument(
        None, help="Attribute name(s) — print just those values, un-elided "
                   "(up to 256 elements each)."),
):
    """Full prim introspection: attributes, relationships, bounds, material binding."""
    payload: dict = {"ref": ref}
    if attr:
        payload["attr"] = list(attr)
    _run("properties", payload)


@app.command()
def bounds(ref: Optional[str] = typer.Argument(
        None, help="@ref or /path (omit: whole stage).")):
    """World-space bounding box one-liner (min/max/center/size)."""
    _run("bounds", {"ref": ref})


@app.command()
def visibility(ref: str = typer.Argument(..., help="@ref or /path.")):
    """Computed + authored visibility — and which ancestor hides a hidden prim.
    Read-only; change with show/hide (isolate --restore for isolate's opinions)."""
    _run("visibility", {"ref": ref})


@app.command("material-binding")
def material_binding(ref: str = typer.Argument(...)):
    """Report a prim's bound material (direct/inherited), or inspect a material @ref's
    shader parameters (MDL/OmniPBR/UsdPreviewSurface)."""
    _run("material-binding", {"ref": ref})


@app.command()
def subsets(
    ref: str = typer.Argument(..., help="Mesh (or subtree) @ref/path."),
    family: Optional[str] = typer.Option(None, "--family",
                                         help="Only this subset family (default: all; "
                                              "coverage is validated for materialBind)."),
):
    """List + validate GeomSubset face assignments (per-face material binding)."""
    _run("subsets", {"ref": ref, "family": family})


@app.command()
def validate(
    fix: bool = typer.Option(False, "--fix", help="Apply the safe repairs (invalid binding "
                                                  "targets, subset family metadata)."),
):
    """Validate the stage: composition, material bindings, subset families, unresolved
    assets, invalid shaders, and renderer compatibility."""
    _run("validate", {"fix": fix})


@app.command()
def isolate(
    refs: Optional[List[str]] = typer.Argument(
        None, help="Refs to isolate (default: selection)."),
    restore: bool = typer.Option(
        False, "--restore",
        help="Remove the visibility opinions isolate authored (do this before "
             "saving a deliverable — isolation is a temporary view)."),
):
    """Show only these prims (+ ancestors); hide the rest. Visual, undoable.
    `--restore` cleanly removes the authored opinions afterwards."""
    _run("isolate", {"refs": refs, "restore": restore})


@app.command()
def convert(
    source: str = typer.Argument(..., help="Source asset (.urdf/.mjcf/.obj/.step/.usd…)."),
    output: Optional[str] = typer.Argument(
        None, help="Output USD path (.usd/.usda/.usdc, or .usdz — packaged with its "
                   "dependencies)."),
    output_format: Optional[str] = typer.Option(None, "--output-format", help="usd|usda|usdc|usdz."),
):
    """Convert a source asset to USD via an external converter (USD inputs pass
    through; USD→USD output goes through the same verified-atomic publish as save)."""
    _run("convert", {"source": _abspath(source), "output": _abspath(output),
                     "output_format": output_format})


# ── physics authoring ──────────────────────────────────────────
physics_app = typer.Typer(help="USD physics schema authoring + validation.", no_args_is_help=True)
app.add_typer(physics_app, name="physics")


@physics_app.command("inspect")
def physics_inspect(root: Optional[str] = typer.Option(None, "--root", help="Subtree @ref/path.")):
    """List mesh candidates with bounds, material, and existing physics schemas."""
    _run("physics.inspect", {"root": root})


@physics_app.command("topology")
def physics_topology(root: Optional[str] = typer.Option(None, "--root")):
    """Inspect authored rigid bodies, colliders, joints, and articulation roots."""
    _run("physics.topology", {"root": root})


@physics_app.command("apply")
def physics_apply(
    file: Optional[str] = typer.Option(None, "-f", "--file", help="JSON operations patch (or '-' for stdin)."),
):
    """Apply explicit low-level physics operations.

    ``scene_paths``, rigid-body/collider/material targets, and bindings are
    caller-selected. Workflows translate higher-level decisions before calling
    this command; usd-cli never infers targets or preservation behavior.
    """
    import json
    import sys
    raw = (sys.stdin.read() if file in ("-", None) else __import__("pathlib").Path(file).read_text())
    patch = json.loads(raw)
    if not isinstance(patch, dict):
        raise typer.BadParameter("physics operations patch must be a JSON object")
    _run("physics.apply", {"operations": patch})


@physics_app.command("validate")
def physics_validate():
    """Deterministic schema validation (mass/density finite, no nested rigid bodies, etc.)."""
    _run("physics.validate", {})


@physics_app.command("simulate")
def physics_simulate(
    scene: str = typer.Option(..., "--scene", help="Pre-authored simulation USD."),
    body: str = typer.Option(..., "--body", help="Exact simulated body @ref/path used to author the recording."),
    body_pattern: Optional[str] = typer.Option(None, "--body-pattern", help="Optional ovphysx solver binding pattern; defaults to --body."),
    rest_position: str = typer.Option(..., "--rest-position", help="Metric rest position as X,Y,Z."),
    world_up: str = typer.Option(..., "--world-up", help="Metric up direction as X,Y,Z."),
    engine: str = typer.Option(
        "ovphysx",
        "--engine",
        help=(
            "ovphysx (real solver; local on supported Linux or Windows hosts, "
            "otherwise through the managed OVRTX adapter when it advertises "
            "physics support)."
        ),
    ),
    duration: float = typer.Option(1.0, "--duration", help="Sim duration (s)."),
    fps: int = typer.Option(30, "--fps", help="Trajectory sample rate."),
    dt: float = typer.Option(1.0 / 240.0, "--dt", help="Solver timestep (s)."),
    output: Optional[str] = typer.Option(None, "-o", "--output"),
):
    """Simulate an explicit scenario → time-sampled recording.usda + raw metrics.

    Play the result with `render-frames --scene <recording.usda>`."""
    def vector(value: str) -> list[float]:
        try:
            result = [float(item.strip()) for item in value.split(",")]
        except ValueError as exc:
            raise typer.BadParameter("must be comma-separated numbers") from exc
        if len(result) != 3:
            raise typer.BadParameter("must contain exactly three values")
        return result
    _run("physics.simulate", {"scene": _abspath(scene), "body": body,
                               "body_pattern": body_pattern,
                               "rest_position": vector(rest_position), "world_up": vector(world_up),
                               "engine": engine, "duration": duration, "fps": fps,
                               "dt": dt, "output": output})


# ── batch & composition ────────────────────────────────────────────────────
# NOT YET IMPLEMENTED — hidden from the CLI until the Session methods exist.
#   * chain  → session.chain()  (missing). Note: History already has the transaction
#              primitives (begin/commit/rollback) that `chain --atomic` would drive.
#   * batch  → session.batch()  (missing). A working `/batch` HTTP endpoint exists in
#              usd_server/app.py, but the CLI would POST to `/cmd`, not `/batch`.
#   * eval   → session.eval()   raises NotImplementedError (AST-sandboxed code-as-action).
# Re-enable each command as its Session method lands.
#
# @app.command()
# def chain(
#     file: Optional[str] = typer.Argument(None, help="JSON array, or '-'/omit for stdin."),
#     atomic: bool = typer.Option(False, "--atomic",
#                                 help="One undo transaction; roll back all on first error (state-model.md §6)."),
# ):
#     """Run commands sequentially from stdin/file. Continue-on-error unless --atomic."""
#     import sys
#     raw = sys.stdin.read() if file in (None, "-") else __import__("pathlib").Path(file).read_text()
#     _run("chain", {"commands": raw, "atomic": atomic})
#
#
# @app.command()
# def batch(file: Optional[str] = typer.Argument(None, help="JSON array, or '-'/omit for stdin.")):
#     """Run commands via the HTTP batch endpoint.
#
#     Reads/renders run in parallel; writes are serialized (USD stage authoring is not
#     thread-safe — state-model.md §6).
#     """
#     import sys
#     raw = sys.stdin.read() if file in (None, "-") else __import__("pathlib").Path(file).read_text()
#     _run("batch", {"commands": raw})
#
#
# @app.command("eval")
# def eval_(
#     code: Optional[str] = typer.Argument(None, help="Python over the ref namespace."),
#     file: Optional[str] = typer.Option(None, "-f", "--file", help="Read code from a file."),
# ):
#     """Code-as-action escape hatch for novel compositions (cli-design §0.8).
#
#     AST-sandboxed; refs (n, m, …) and the manipulation/spatial verbs are in scope.
#     """
#     if file:
#         code = __import__("pathlib").Path(file).read_text()
#     elif code in (None, "-"):
#         import sys
#         code = sys.stdin.read()
#     _run("eval", {"code": code})


# ── feedback (client-side; bypasses dispatch) ───────────────────────────────
_REPORT_ISSUE_TEXT = """\
Report a bug or request a feature — https://github.com/NVIDIA-Omniverse/usd-content-agents/issues

If you are an AI agent using usd-cli and you hit a bug, a crash, wrong or surprising
output, or a capability you needed but couldn't find: please file a GitHub issue.
Do it when you encounter the problem — don't just silently work around it. Then
continue the task with a workaround if one exists.

How to file
  1. Search for an existing issue first:
       gh issue list --repo NVIDIA-Omniverse/usd-content-agents --search "<keywords>"
     If one matches, add a comment with your details instead of opening a duplicate.
  2. File (write the body to a file first; use --label bug or --label enhancement,
     dropping --label if the repo rejects it):
       gh issue create --repo NVIDIA-Omniverse/usd-content-agents --label bug --title "<one-line summary>" --body-file /tmp/issue.md
  3. No `gh` or no GitHub credentials? Show the user the title + body you prepared
     and ask them to submit it at https://github.com/NVIDIA-Omniverse/usd-content-agents/issues/new

Bug report body — include:
  - Environment: {env}
  - Exact command(s) run and their full output (re-run with --json for the
    structured envelope; include the error/traceback verbatim)
  - Expected vs actual behavior
  - Minimal repro: the smallest scene + command sequence that triggers it
    (inline a small .usda if possible)

Feature request body — include:
  - The task you were trying to accomplish (goal, not just the missing flag)
  - What you tried and where it fell short
  - Proposed CLI surface, if you have one
"""


@app.command("report-issue")
def report_issue():
    """Show how to file a bug report or feature request as a GitHub issue.

    Agents: if a usd-cli command errors, renders wrong output, or lacks a capability
    you needed, run this and file an issue at
    https://github.com/NVIDIA-Omniverse/usd-content-agents
    — then keep working around it."""
    import platform
    from importlib.metadata import PackageNotFoundError, version

    from usd_core import load_config
    from usd_core.render.factory import resolved_renderer

    cfg = load_config()
    try:
        ver = version("usd-cli")
    except PackageNotFoundError:  # source checkout without an install
        ver = "0+unknown"
    env = (f"usd-cli {ver} | engine: {cfg.backend.get('engine')} | "
           f"renderer: {resolved_renderer(cfg)} | {platform.platform()}")
    typer.echo(_REPORT_ISSUE_TEXT.format(env=env))


_GLOBAL_FLAGS = {"--json", "-q", "--quiet"}


def _hoist_global_flags(argv: list[str]) -> list[str]:
    """Allow --json/-q/--quiet anywhere on the command line.

    Typer only accepts them before the subcommand; agents habitually type
    `usd-cli snapshot --json`. Hoist bare global flags to the front so both
    placements work. Anything after `--` is left untouched.
    """
    head: list[str] = []
    tail: list[str] = []
    passthrough = False
    for tok in argv:
        if tok == "--":
            passthrough = True
            tail.append(tok)
        elif not passthrough and tok in _GLOBAL_FLAGS:
            head.append(tok)
        else:
            tail.append(tok)
    return head + tail


def run() -> None:
    _configure_windows_text_streams()
    invoked = os.path.basename(sys.argv[0] or "")
    for alias in ("ov", "3dsc"):
        if invoked == alias or invoked.startswith(alias + "."):
            print(f"note: `{alias}` is now `usd-cli` (project renamed); "
                  f"the `{alias}` alias will be removed in a future release.",
                  file=sys.stderr)
            break
    sys.argv[1:] = _hoist_global_flags(sys.argv[1:])
    try:
        app()
    except ValueError as exc:  # argument parsing (vec etc.) — one clean line
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    run()
