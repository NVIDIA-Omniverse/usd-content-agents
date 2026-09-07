# Hard-Local Worker Policy

Read this reference for a classified local hole, generated-patch remesh,
two-part intersection, or unavailable PMP, WildMeshing, or MCUT adapter.

## Fixed Route Order

Use `select_hard_mesh_routes` only after the defect, protected boundaries, and
attribute support are explicit:

| Defect | First route | Candidate/shadow route | Last resort |
| --- | --- | --- | --- |
| Classified 3/4-vertex hole | enabled bounded Trimesh fill | PMP candidate, WildMeshing shadow | opt-in SDF reconstruction |
| Explicitly classified 5+ vertex accidental hole | enabled PMP fill | WildMeshing shadow | opt-in SDF reconstruction |
| Generated patch quality outside classified-hole fill | unavailable | WildMeshing shadow | opt-in SDF reconstruction |
| Two named intersecting parts | enabled Geogram | MCUT shadow | opt-in SDF reconstruction or reject |

No local mutation is eligible without explicit defect intent, frozen/protected
boundaries, and supported attributes. Shadow routes never publish production
authority.

## Adapter Contract

`run_hard_mesh_executable` accepts only a pinned `HardMeshExecutableSpec`, fixed
JSON request/report schemas, a bounded timeout, inherited memory limits, and an
assigned work directory. It verifies:

- capability, executable, source-tree, request, source, output, and auxiliary
  artifact identities;
- exact worker, operation, implementation version, and build ID;
- changed/generated face IDs, frozen vertices, protected edges, envelopes,
  correspondence, and attribute-transfer artifacts;
- path containment, digest integrity, bounded report size, and immutable input
  preservation.

Keep `success`, `unavailable`, `refused`, and `failed` distinct. A projected
legacy `WorkerResult` may map refusal to unavailable, but the richer hard-mesh
report remains authoritative.

## Current Availability

- `pmp_patch.py` implements typed classified-hole filling for an explicit 5+
  vertex accidental-hole loop.
- `wildmeshing_shadow.py` implements a shadow-only invariant/rollback envelope
  contract.
- `mcut_shadow.py` implements a shadow-only two-part fragment and intersection
  curve inventory; fragment selection, deletion, and union are prohibited.

PMP is registry-enabled and exported through `worker_runner` only for that
classified-hole operation. Build its pinned native helper outside the
repository, then supply both `GEOMETRY_REPAIR_PMP_EXECUTABLE` and
`GEOMETRY_REPAIR_PMP_EXECUTABLE_SHA256`; discovery independently hashes the
binary. Missing or mismatched identity is `unavailable`, not repaired. Inferred
hole intent, generic remeshing, and global reconstruction remain unauthorized.

WildMeshing and MCUT are not registered production workers or exported by
`workers.__init__`. Report them as `unavailable`, not repaired or
production-ready.

The reconstructive SDF route uses `sdf_tools` and an explicit driver admitted by
checked-in `sdf_tools` policy, then separately behaviorally qualified by
`geometry_repair/sdf_backend_qualifications.json`. OpenVDB 13 is the current
qualified driver. An agent may select its qualified ID but may not install,
import by module path, or register another backend. SDF driver execution is
in-process within the existing generic worker boundary.

## Promotion Requirements

Require registry approval, exact binary/source pinning, transitive-license and
distribution review, a native-closure gate that rejects any introduced or
bundled LGPL component, packaged execution, deterministic repeated hashes,
resource-limit tests, correspondence coverage, protected-feature preservation,
and benchmark evidence before granting candidate or production authority. A
successful native return still requires independent re-diagnosis and fidelity.
