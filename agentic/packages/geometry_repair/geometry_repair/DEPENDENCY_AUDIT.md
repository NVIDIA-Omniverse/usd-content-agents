# Geometry Repair Executable Dependency Audit

Audit snapshot: 2026-08-13 UTC.

This file records the executable third-party boundary used by the Geometry
Repair package. It is not a substitute for NVIDIA distribution-compliance
review. `worker_registry.json` is the runtime authority: code that is absent,
disabled, unpinned, or outside the approved open-source/NVIDIA classes cannot
be selected.

The public Geometry Repair distribution contains no native B-rep backend.
STEP, IGES, and BREP inputs require a separately authorized runtime, or the
authoring provider must include validated USD or mesh geometry with explicit
units and axes. The public locks, wheels, containers, and staged source exclude
native OCP bindings.

## CoACD Collision Worker

- Python package: `coacd==1.0.11`.
- Upstream source: `SarahWeiii/CoACD`, tag `1.0.11`, commit
  `b678aa0802996fa03e1ec0e68bd05acf8cd20cf9`.
- Primary license: MIT, copyright Xinyue Wei and Minghua Liu.
- Runtime enforcement: the isolated runner refuses any installed version other
  than `1.0.11`; the coordinator records the version, requested threshold,
  deterministic seed, output hashes, logs, elapsed time, and all hulls.

The audited source build references only open-source components:

| Component | Pinned source in CoACD | License family | Runtime relevance |
| --- | --- | --- | --- |
| CDT | `ec03b309fd18102ab1da069f2edf3b37be5d1fb3` | MPL-2.0 | Triangulation |
| OpenVDB | `v8.2.0` | MPL-2.0 | Bundled CoACD preprocessing support |
| oneTBB | `v2022.0.0` | Apache-2.0 | OpenVDB parallel runtime |
| Boost | `1.81.0` | Boost Software License 1.0 | OpenVDB support |
| spdlog | `v1.8.2` | MIT | Native logging |
| zlib | `v1.2.11` | zlib | Compression support |
| Eigen | `3.4.0` when fetched | MPL-2.0 | Native linear algebra |

The CDT Catch2 submodule is test-only and is not invoked by the production
worker. The installed wheel carries CoACD's MIT license in its distribution
metadata. Any redistribution of the native wheel must also preserve the
applicable transitive notices and source-availability obligations, especially
MPL-2.0 covered files.

## Other Enabled Third-Party Workers

- Current OpenVDB SDF driver: the repository-built `openvdb==13.0.0+wu.3` wheel
  uses OpenVDB 13.0.0 source commit
  `7c03e1f084873cd1b3422c7ff7aec6ee681b3b38`. Its source lock records the
  immutable archives and source trees for OpenVDB, nanobind, robin-map,
  oneTBB, c-blosc, and zlib, plus the overlay, patch, redistribution-file, and
  pinned build-tool versions. Runtime admission checks ABI 13, both installed and
  source-lock distribution versions, the exact commit, and loaded-extension
  SHA-256 evidence. OpenVDB is Apache-2.0; the wheel retains OpenVDB's license
  plus the permissive notices for nanobind, robin-map, oneTBB, c-blosc, zlib,
  LZ4, Zstandard, Bitshuffle, FastLZ-derived code, and zlib-ng-derived code.
  Snappy is disabled and absent. c-blosc, zlib, and their codec inputs are
  linked statically; source-built oneTBB is the only non-platform shared
  dependency and is bundled by `auditwheel`.
  Geometry Repair calls the backend-neutral `sdf_tools` contract inside its existing
  resource-limited generic worker subprocess; there is no OpenVDB-specific
  helper, daemon, or nested process boundary. The admitted `openvdb_runtime`
  driver invokes OpenVDB's nanobind extension and bounded tools overlay in that
  importing CPython process. It operates on one USD mesh prim at a
  time, refuses attributes it cannot remap, and records runtime identity,
  controls, resource use, and canonical input/output arrays for each
  reconstruction. Independent source-relative drift, topology,
  protected-feature, and correspondence gates remain authoritative. The
  `sdf_collision_rebuild` registry entry is not another dependency: it
  grants collision-only authority to the same wheel and facade. Its neutral
  world-space bridge bypasses render-attribute transfer only because render USD
  is immutable and independently hash-checked; it cannot promote a render
  repair. The GPL evaluation helper is not compiled or shipped in production.
- Geogram/vorpalite: BSD-3-Clause, pinned to release `v1.10.0` at commit
  `c8529bb00838186938ab31d96008a59b6a892dee`. This release supplies the
  `Linux64-nonx86-gcc-dynamic` platform needed by the ARM64 image; the prior
  v1.8.5 release only supplied an x86-oriented `-m64` Linux configuration. The
  container build verifies the
  detached Git commit and recursively checks out the submodule commits recorded
  by that tree; commit archives are not used because they omit required
  `libMeshb` sources. Graphics/Lua/TetGen/Triangle/legacy numeric modules are
  disabled. The reviewed source patch makes `GEOGRAM_WITH_TBB=OFF` disable the
  release's otherwise-unconditional Linux parallel-STL link, so Geogram uses
  sequential STL and adds no oneTBB build or runtime dependency. Only the native
  repair executable and its required library are built. Runtime selection
  independently hashes the executable against
  `GEOMETRY_REPAIR_GEOGRAM_EXECUTABLE_SHA256` or its read-only `_FILE`
  equivalent, then refuses any approved binary that does not report
  `vorpalite 1.10.0`. The verified digest is retained in worker evidence. The
  worker disables hole filling, component deletion, and internal-shell removal.
  Exact submodule commits and permissive licenses are recorded in
  `GEOGRAM_THIRD_PARTY_NOTICES.md`; the image retains that notice, the recursive
  submodule status, Geogram's BSD license, and discovered upstream license
  files under `/opt/geogram/share/licenses`.
- Trimesh: MIT. Runtime version is captured for cleanup and convex hull output.
- NetworkX: BSD-3-Clause. Trimesh uses it to reconstruct scene/object graphs for
  graph-bearing formats such as 3MF; it is intake infrastructure and does not
  mutate geometry.
- lxml: BSD-3-Clause. Trimesh uses its bounded XML parser for 3MF package
  intake; Geometry Repair does not enable network-backed XML resolution.
- fast-simplification `0.1.13`: MIT. It is used only for collision-role working
  geometry: either to create a bounded CoACD input after watertightness,
  winding, p99 surface-drift, and volume-drift checks, or to reduce an
  over-budget convex candidate before the complete volume, occupancy,
  surface-distance, protected-feature, complexity, and runtime gates run.
  Render geometry is never replaced by this output.
- Rtree: MIT wrapper over libspatialindex (MIT), used for deterministic
  closest-surface and inside/outside queries. It is measurement authority, not
  a mutating repair worker.
- Embreex `4.4.0`: BSD-3-Clause Python binding over Intel Embree
  (Apache-2.0), used to keep bounded inside/outside collision probes tractable
  on high-triangle rigid assets. It is a measurement accelerator only.
- Pillow: HPND, used only for software orthographic silhouette evidence.
- SciPy/Qhull: BSD/permissive dependencies used by the Trimesh convex hull path.
- OpenUSD 25.5, supplied only by NVIDIA `usd-exchange==2.3.0`: Tomorrow Open
  Source Technology License 1.0 (`LicenseRef-TOST-1.0`), an Apache-derived but
  materially modified license, used for stage inspection, official compliance
  validation, and authoring. The USD Exchange SDK wrapper is Apache-2.0 and its
  wheel retains `usd-license.txt` plus its other native dependency notices.
  The CAD Agent app and service images pin
  `python:3.12.13-slim-bookworm` by manifest digest
  `sha256:8a7e7cc04fd3e2bd787f7f24e22d5d119aa590d429b50c95dfe12b3abe52f48b`.
  This is an executable compatibility boundary: the same 25.5 manylinux wheel
  aborts on import under the Debian 13/glibc 2.41 base selected by the
  unqualified `slim` tag, while the pinned Debian 12/glibc 2.36 image passes
  isolated `UsdUtils.ComplianceChecker` execution.
  Local CAD/geometry packages must not additionally depend on `usd-core`:
  `usd-core` and `usd-exchange` both own `pxr/*`, and concurrent installation
  was observed to create nondeterministic mixed shared objects and import-time
  aborts. `tests/test_pyproject_usd_dependencies.py` enforces the single-provider
  boundary.
  The complete upstream license and notices must be retained; it must not be
  represented as SPDX `Apache-2.0`.
- Khronos glTF Validator: official Apache-2.0 command-line validator pinned to
  pre-release `2.0.0-dev.3.10`. Both CAD Agent production images download the upstream
  Linux archive and requires SHA-256
  `168eba887964125abe17ae97899b38d0b3cfd73c266c78424c194929ddcbc522`;
  the archive's `LICENSE` and `NOTICES` remain under `/opt/gltf-validator`.
  This precompiled archive is x86-64-only and the image stage fails explicitly
  on other architectures rather than running an incompatible binary.
  Other deployments discover only the exact `gltf_validator` executable or
  use `GEOMETRY_REPAIR_GLTF_VALIDATOR_EXECUTABLE`; absence is reported as
  `not_evaluated` and never replaced by a non-authoritative parser pass.
- PMP Library classified-hole filling: MIT, pinned to commit
  `2a2ad502743724ba90e09af816364c84032f9015` (upstream version
  `3.0.0`). The independently pinned Git tree-inventory SHA-256 is
  `ec134774c578b2ddd97b620daaa73ee1839ffb78b9d3355050443d8bdb6e5508`.
  Only the repository-owned `geometry_repair_pmp_patch` wrapper and PMP's
  non-viewer core are built; examples, viewers, tests, Polygonal, regressions,
  OpenMP, and shared libraries are disabled. The compiled operation fills one
  caller-classified complete boundary loop of at least five vertices. It does
  not select holes, delete source topology, or expose generated-patch
  remeshing as a successful operation.
  `scripts/build_pmp_patch.sh` checks out and verifies sources outside Git,
  builds outside the repository, installs outside the repository, emits a
  build manifest, and copies third-party notices. The Python production path
  independently hashes the executable and requires the approved digest in
  `GEOMETRY_REPAIR_PMP_EXECUTABLE_SHA256`; a matching self-reported build ID is
  insufficient. A clean checkout already at the approved commit is reused
  without a remote fetch, and source-directory/repository environment
  overrides support hash-verified CI caches or authorized mirrors without
  changing the pin. Capabilities and every native report retain the executable
  SHA-256, PMP commit, and PMP tree-inventory SHA-256.

  | PMP build dependency | Pin | License | Shipped use |
  | --- | --- | --- | --- |
  | pmp-library | `2a2ad502743724ba90e09af816364c84032f9015` | MIT | Hole triangulation/refinement/fairing |
  | Eigen | vendored PMP `external/eigen-3.4.0` | MPL-2.0 plus retained upstream notices | PMP linear algebra |
  | nlohmann/json | `9cca280a4d0ccf0c08f47a99aa71d1b0e52f8d03`; tree inventory `2a9391f4fb9775e87ae994395672831d29d8496026a536031f6deab453cd07a8` | MIT | Strict native protocol JSON |
  | OpenSSL libcrypto | distro `libssl-dev`; build manifest captures exact version | Apache-2.0 | Native artifact SHA-256 |

  The neutral OBJ bridge contains only the exact named local-space triangle
  mesh. Python retains ownership of USD composition, hierarchy, material
  binding, custom attributes, immutable source faces, and final authoring.
  Face subsets, face-varying data, authored UVs/normals, instances, non-rigid
  transforms, non-triangular source faces, incomplete loops, or unapproved
  executable digests fail closed.

Scene Optimizer, OVRTX, ovphysx, PhysX/Isaac runtime checks, and SimReady
validators are NVIDIA first-party boundaries and are not reclassified as
third-party dependencies by this package. Scene Optimizer is used both after
repair and, with split/merge/deduplication disabled, as a typed isolated
de-instancing prerequisite. Its output must independently preserve world-space
mesh arrays or the exact nondegenerate surface set before another operation may
advance.

## Optional Portability Evaluation

- MuJoCo `3.3.7`: Apache-2.0. It is an optional comparison runtime, not the
  NVIDIA certification authority or a geometry mutation worker.
- PyBullet `3.2.7`: zlib license. It is an optional comparison runtime under
  the same boundary. Simulator-specific convex cooking and contact differences
  are retained in evidence and never substituted for `ovphysx` results.

## Explicitly Non-Executable

CGAL/libigl hard-local repair, CelloCut, PaMO, PMP generated-patch remeshing,
and other research candidates remain disabled until their exact implementation,
version, transitive licenses, distribution model, and bounded worker contract
are approved. The CGAL exact
intersection binary may be compiled only through the opt-in local evaluation
target; it is not included in wheels or production images and cannot contribute
certifying evidence. Source availability alone does not enable a worker.
