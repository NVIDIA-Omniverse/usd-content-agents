# Geometry Repair Dependency Policy

Geometry Repair may execute only:

1. code owned and authorized by NVIDIA for this workflow; or
2. third-party software with a verified open-source license, pinned source or
   package version, recorded transitive dependencies, and an approved
   distribution model.

Public source without a license, source-available/non-commercial terms, remote
commercial services, and proprietary third-party SDKs are not eligible.

The executable registry is `geometry_repair/worker_registry.json`. A worker is
callable only when its status is `enabled` or `enabled_with_notice`, and only
for an exact operation listed in that worker's registry entry. Runtime worker
objects must declare their operation set, and startup/dispatch checks require
that set to be a subset of the registry authorization. NVIDIA
first-party workers marked `enabled_external` remain behind their owning shared
workflow boundary. Copyleft or mixed-license workers remain disabled until the
exact modules, linking model, notices, source obligations, and deployment model
are approved.

No native B-rep backend is distributed with the public Geometry Repair runtime.
STEP, IGES, and BREP inspection or healing require a separately authorized
backend. Without one, diagnosis fails closed and directs callers to provide a
validated USD or mesh representation from the authoring provider.

Every enabled worker must record its implementation version, deterministic
seed where applicable, exact worker and invoked operation, and output digest in the repair
attempt ledger or typed collision report. A worker's successful exit is never
repair acceptance; the independent topology, fidelity, identity, and profile
gates remain authoritative.

`coacd_collision` is pinned to `coacd==1.0.11` and is the only enabled
approximate convex decomposition worker. Its MIT/MPL/permissive transitive
review is recorded in `geometry_repair/DEPENDENCY_AUDIT.md`. The runner is
isolated, version-enforced, seeded, resource-limited, and its output paths,
digests, convexity, watertightness, winding, hull count, vertex count, and
volume error are independently checked. An upstream warning that the requested
concavity threshold was not achieved remains a conditional workflow warning.

`sdf_rebuild` is repository-owned Apache-2.0 orchestration over an admitted SDF
backend. Its current `openvdb` driver uses OpenVDB 13.0.0 at commit
`7c03e1f084873cd1b3422c7ff7aec6ee681b3b38` through the source-locked
`openvdb==13.0.0+wu.3` wheel. Runtime admission requires ABI 13, matching wheel
and source-lock distribution identities, the exact source commit, and the
loaded extension's path and SHA-256 evidence. Production containers retain the
OpenVDB license and the permissive notices for bundled native dependencies.
Content code calls the repository's backend-neutral `sdf_tools` contract. Its
admitted OpenVDB driver uses the `openvdb_runtime` facade, which invokes
OpenVDB's nanobind Python extension with the repository's bounded tools overlay
in the importing process. It does not launch an OpenVDB helper, daemon, or
subprocess. The separately compiled GPL CGAL evaluator is never part of the
application build, wheel, runtime image, or certification path.
`sdf_collision_rebuild` is separate logical policy
authority over the same wheel and facade. It may author only a neutral
collision working copy and collision layer, requires explicit reconstructive
opt-in, records render SHA-256 identity, and remains conditional after native
success.

Mutation jobs execute in the generic Geometry Repair isolated worker under the
request's wall-clock and address-space budgets. OpenVDB operations execute
in-process inside that worker through `openvdb_runtime`; the generic worker
boundary is not an OpenVDB-specific adapter process. The no-op source
checkpoint remains the only candidate operation executed in the coordinator.
