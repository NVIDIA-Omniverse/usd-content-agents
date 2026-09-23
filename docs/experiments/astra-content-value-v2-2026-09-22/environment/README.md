# Shared environment for the controlled rerun

Both arms use the same e640b8d6aa667745830db5e9bf87fd1b4cd763cb implementation,
Python dependencies, native tools, rendering/physics runtimes, sources and
budgets. Workflow use is the treatment. This is a new implementation/version
study; the frozen v1 scores are unchanged. Setup and import readiness are not
end-to-end workflow qualification.

## Confirmed v1 setup gaps

The public setup script fetches Scene Optimizer by default. V1 omitted that
resource; its repair route also searched the author's cwd instead of the
supervisor repo. Always set WU_SO_PACKAGE_DIR explicitly. The checked archive
here is public Scene Optimizer1.0.3, USD25.11/Python3.12/x86_64, SHA256
9d98d22eed1eb31da3183bfd4155f3b8eca48576e6eb5947d126e781f0edc671.
It runs in its own ABI-isolated subprocess; do not add its pxr to the main venv.

Geogram is an optional separately admitted repair backend. Production routing
requires pinned vorpalite1.10.0, exact upstream commit/submodule provenance,
the repository's no-TBB patch, and independently approved executable SHA256.
The build script does not automatically approve a new binary. Copy the same
qualified binary/library closure to the second host; separate builds may differ
in timestamps/toolchain bytes. Never modify worker_registry.json to enable a
blocked route. PMP and OpenVDB optional availability is recorded separately;
missing optional routes are not silently treated as available. If the frozen
protocol permits a route it needs its own successful setup and synthetic probe.

The base setup must keep usd-exchange as sole pxr owner. usd-core and
usd-exchange must not coexist in the main venv. Native source validation must
actually run. The pinned usd-exchange2.3.0 provider omits pxr.UsdValidation:
the repository compatibility shim explicitly skips extension-dependent rules.
The supported standalone NVIDIA custom rules execute. This declared limitation
is identical in both arms; neither import/help nor wrapper success establishes
full registered-rule coverage. Do not silently substitute another pxr provider.
Source format/geometry/collision roles remain separate. The e640 implementation
includes source-preserving glTF transfer, repair index fix, mounted Physics
runtime, typed physical-evidence consumer and corrected8..64 cook-option range.
Those are code changes applied equally to both arms, not evidence of generic
simulation readiness or a clean causal interpretation of v1.

## Install identical hosts

Root chooses the host and provisions its GPUs. Run only in a fresh new root:

```sh
nice -n 15 bash environment/launch_bootstrap.sh /opt/astra-content-value-20260922-rerun
```

This uses a real shallow Git fetch/checkout of the exact public commit (not an
archive without .git), verifies setup/lock hashes, verifies/extracts the public
Scene Optimizer archive, runs standard setup, installs CAD0.2.0/OCP7.8.1.1/
VTK9.3.1/rtree, and checks dependency consistency and skill symlinks. No credentials
are copied. UV_NO_CACHE and TMPDIR remain in the new root. The script stops at
23GiB /opt usage, reserving headroom below the25GiB ephemeral quota; df's host
filesystem capacity is not the relevant quota. Some staging installs can peak
between checks, so root must monitor du during runtime downloads.

Copy the resulting environment/evidence/thirdparty.constraints.txt to host2 and
supply it from its first install (uv's UV_CONSTRAINT propagation is verified):

```sh
nice -n 15 bash environment/launch_bootstrap.sh /opt/astra-content-value-20260922-rerun \
  --constraints /opt/astra-content-value-20260922-rerun/environment/thirdparty.constraints.txt
nice -n 15 bash environment/build_geogram.sh /opt/astra-content-value-20260922-rerun
```

After independent Geogram provenance/binary review, root supplies the exact
approved hash. It is never inferred from a successful task:

```sh
python3 environment/emit_environment.py --root /opt/astra-content-value-20260922-rerun \
  --approved-geogram-sha256 APPROVED_SHA256 > environment/shared.env
. environment/shared.env
repo/.venv/bin/python environment/provision_native.py --root /opt/astra-content-value-20260922-rerun
repo/.venv/bin/python environment/record_runtime.py --root /opt/astra-content-value-20260922-rerun \
  --output environment/evidence/runtime.json
```

provision_native uses the pinned repository's own exact-lock installers and
constructor/import probes. No scenes, steps, models or renders are run.
OvPhysX0.4.13 and OVRTX0.4.1.364340/ovstage0.1.1.355824/Warp1.16.0 use separate venvs.
The main constraints file must not be applied to those independent lock installs.
Run compare_environments.py on both inventories; it rejects differences in
code, Python, packages, native file hashes, locks, skills and repair availability.
Runtime equality does not establish workload isolation; root's per-run mount,
process-visibility, resource and simulator-concurrency harness must enforce that.

The source package-lock remains the original e640 file. The recorded installed
dependency override is @openai/codex-sdk0.154.0 and @openai/codex0.154.0, including
its matching codex-linux-x64 binary package. Both hosts must apply this same
override without saving or rewriting package-lock.json. Bootstrap does so with
`npm install --no-save --package-lock=false`. record_runtime.py hashes all three
installed packages; the child SDK and executable must both be pinned.

## Read-only namespace tools

export_namespace_tools.py creates a new hardlinked export; generated files are
replaced through new inodes before relocation so the original installed files
remain byte-exact. It includes the Python base, three venvs, source-only real Git,
Node dependencies, SceneOptimizer and Geogram. Editable paths, console launchers,
Python's generated sysconfig library locator and native readiness markers point
to /tools. No implementation source is patched. Staging directory modes are
made readable on the separate export directory inodes. The historical random
venv-creation command is removed from pyvenv.cfg; it is not a runtime setting.

On both hosts, after native provision and the same Geogram archive extraction:

```sh
nice -n 15 ionice -c3 repo/.venv/bin/python environment/export_namespace_tools.py \
  --root /opt/astra-content-value-20260922-rerun \
  --output /opt/astra-content-value-20260922-rerun/namespace_tools_v5 \
  --receipt /opt/astra-content-value-20260922-rerun/environment/evidence/namespace_tools_export_r5.json
nice -n 15 repo/.venv/bin/python environment/run_namespace_smoke.py \
  --tools /opt/astra-content-value-20260922-rerun/namespace_tools_v5 \
  --output /opt/astra-content-value-20260922-rerun/environment/evidence/namespace_smoke_06
```

Paths are create-only; use a fresh numbered attempt on failure. The export's
environment.json is the shared environment map. Both AUTO_PROVISION values are0.
Mount the export read-only at /tools, use a fresh writable HOME, and provide a
private /dev/shm with mode1777 (`--perms 1777 --tmpfs /dev/shm`) before dropping
UID/capabilities. A default root-owned tmpfs mode fails native named semaphores.
Root bwrap must create PID/network/IPC/UTS namespaces without --unshare-user;
then /usr/bin/setpriv drops UID, supplementary groups and all capabilities and
sets no-new-privileges. The native daemon also opens sibling provision locks even with provisioning
disabled. Bind separate lane-owned single-link regular files mode0600 over
/tools/ovphysx-venv.provision.lock and /tools/ovrtx-venv.provision.lock. OVRTX also
needs the qualified, narrowly scoped generated-cache overlays; the bundled
shader cache remains read-only inside them. Code and library files remain
read-only. The experiment harness owns the
stronger cgroup/network/GPU constraints. This environment smoke mounts no author, source or evaluator.

Before native model-backed stages, follow [NATIVE_CHILD_ROUTING.md](NATIVE_CHILD_ROUTING.md).
The shared provider file is generated in each fresh author home; native commands
must explicitly select Astra Ultra and canonical `--repo-root /tools/repo`.
The finalizer now invokes `prepare_native_git_ownership.py` with sudo for only
the exported repository root and its `.git` directory, setting UID/GID20000 to
match every author lane. All file bytes, paths and modes remain unchanged, and
the namespace mounts remain read-only. The native source verifier intentionally
ignores ambient Git safe-directory configuration. Its actual UID20000 test must
pass; ordinary `git status` under a different UID is insufficient.

Independent evaluation uses a separate rootfs prepared by
`prepare_evaluator_rootfs.py` from the qualified author skeleton. It adds only
empty `/original`, `/evaluator`, `/evidence` and `/evaluation_driver.py` targets.
Never add those targets to the author rootfs. The separate qualification checks
read-only workspace/source/tools, zero credentials/capabilities, network denial,
real USD parser success/rejection and unavailable-parser classification.

Node1 namespace_smoke_06 passed10checks: real CLI execution, imports,
source Git, SceneOptimizer output, Geogram output, supported NVIDIA validator,
read-only native runtime resolution and OvPhysX constructor. Earlier attempts
retain mount/permission/library failures. OVRTX lock readiness is not a render.
Use compare_namespace_exports.py on both export receipts. It compares all file
bytes/symlink targets except Git administrative metadata; identical clean source
bytes and the identical derived commit remain required. The complete stronger
namespace and GPU checks are separate from this export parity check.

The small independent native dynamics probe reuses a passing isolated launch:

```sh
nice -n 15 repo/.venv/bin/python environment/run_namespace_native_step.py \
  --qualified-smoke environment/evidence/namespace_smoke_06 \
  --output /opt/astra-content-value-20260922-rerun/environment/evidence/namespace_native_step_02
```

It performs48CPU steps for each of two synthetic cubes. Gravity-on must fall;
gravity-zero must remain still and fail the falling predicate. It does not
qualify any benchmark task, render, model, or complete agentic stage chain.

## Qualification gates before scored authoring

1. Run the repository's AGENTS.md required checks and skill sync on the supervisor
   clone. They must use the actual installed environment. Run the local synthetic
   environment tests too; forged native hash and unequal inventory tests fail closed.
2. Explicitly run smoke_backends.py --execute on a new output directory. It
   exercises real Scene Optimizer, approved Geogram and native USD validation
   against synthetic shapes only. Review any skipped native rule/ABI warning;
   source validator success alone may not mean every registered rule executed.
3. Run content-workflow-cli preflight physics-runtime --repo-root REPO --report
   REPORT. Retain actual lock/constructor readiness, then a small synthetic
   physical step and positive/negative expected outcomes under root's global cap.
4. Run OVRTX render-probe plus an actual at-least640x480 synthetic visible render
   in each of four fresh isolated cwd/session/GPU assignments. Inspect every PNG.
   A tiny blank probe is not readiness. Configure server.allowed_roots and
   server.allowed_write_roots, not the ignored allowed_read_roots key. Per-session
   GPU UUID selection and observed process/device ownership must both be recorded.
5. Exercise actual native workflow children with gpt-6-astra/ultra only and
   verify their tool execution, request admission, terminal usage and automatic
   capture. Record genuine workflow/task failures separately from environment
   availability. No fabricated approval or skipped required task check is allowed.
   Native Geometry is a conditional geometry handoff; physical task acceptance
   additionally requires the independent task evaluator. A model planning exit0
   cannot establish a successful native stage chain.

The retained qualifications distinguish actual four-GPU renders, native child
capture and low-level physics from task success. The q13 native Physics attempt
reported a runtime-readiness failure; its original output remains unchanged.
The follow-up synthetic same-session renderer→CPU-physics probes expose the
normally suppressed constructor output and measure process-limit events. They
are environment diagnostics, not a repaired q13 task result. Use the final
freeze-bound qualification receipts for current status and resource limits.

## Source-only author view and Git provenance

The supervisor retains the exact upstream Git clone and qualifying tests. Do not
mount v1 outputs, evaluator internals, publication worktrees or private reviews in
an author's namespace. e640 predates public experiment bundles but contains a
case-specific capstone diary and top-level regression tests. make_author_repo.py
can create a parentless real Git source snapshot excluding these, with every
retained byte hashed and the upstream commit attested separately. Its commit is
explicitly a derived snapshot, never misreported as e640. Root must freeze this
same view for both arms and mount it at the installed editable-source path;
otherwise editable .pth paths can resolve to the forbidden supervisor clone.
Do not simply remove .git: native lineage/provenance checks need real Git.

The repo's default AGENTS.md routing remains available identically. The plain
arm's explicit experimental instruction overrides default workflow routing;
only the treatment instruction differs. Both arms can access the same installed
low-level tools; neither can inspect the opposite arm, held-out evaluator source
or prior outcomes. Native child process/model limits must be enforced by the
same global harness and included in run accounting.

The actual native_step_02 result passed both controls in3.29s: gravity-on ended
at z0.7997125387191772m; gravity-zero stayed at z1m. Each native trace has48steps
and7samples, with original synthetic scene hashes unchanged. The failed first
step attempt documented the read-only provision-lock incompatibility.

The common single-Astra model catalog is separately provided by root under
SHA256 e612a56f9fef06041483ddd20cbea586f50d2fbf12e558e852dfa40bd332f976.
supplement_namespace_tools.py adds its exact bytes at /tools/astra_model_catalog.json
and the two empty lock mount targets without changing any existing installed
source/runtime byte, then writes a new inventory referencing the unchanged
parent receipt. The catalog content is not included in diagnostic reports or
public documentation. Its exact hash belongs to the common tool profile; model
gateway/request effort qualification remains the supervisor's separate gate.

For a fresh final export, `finalize_namespace_tools.sh ROOT CATALOG_JSON` runs
export → exact catalog supplement → installer metadata normalization → empty
OVRTX generated-cache target preparation → isolated
functional smoke → native positive/negative gravity probes. All outputs are
create-only. Use this complete sequence for reproduction; the numbered attempts
above explain the retained qualification history.

normalize_export_metadata.py removes only uv's editable-install timestamp cache
from the exported environment, replaces random staging names in shell prompts,
and updates affected wheel RECORD entries to the actual relocated file hashes.
The original timestamp-bearing install metadata stays unchanged in the supervisor
installation and earlier retained receipts. Native libraries, implementation,
package versions and upstream source are unchanged. This avoids falsely calling
ignored byte differences equal: compare_namespace_exports.py still demands exact
installed-file parity, excluding only Git administrative data. The same script
also records every removed/changed metadata file and its original installed hash.

Actual isolated GPU qualification exposed another native runtime requirement:
OVRTX writes texture, derived-data and MDL extension caches below its installation.
`prepare_ovrtx_cache_targets.py` adds only the initially absent, empty mode0755
`ovrtx/bin/mdl/omniverse_exts` directory. Both v8 inventories prove installed file
bytes unchanged and equal. The namespace harness supplies exact lane-private
writable generated-cache directories; the existing 865,651,037-byte vendor
shader cache remains separately read-only. No broad writable venv is allowed.
The first cache-corrected GPU attempt also exposed the native thread pool hitting
the fixed512 PID/thread cap; its failure and full native logs remain retained.
The CPU environment receipt is consequently not a claim that GPU rendering or
the native model workflow is qualified. Those gates have separate final receipts.
