# OpenVDB Runtime Driver

`openvdb-runtime` is the OpenVDB 13 implementation driver for the
backend-neutral `sdf_tools` contract. Content-agent and geometry workflow code
selects semantic SDF operations through `sdf_tools`; it does not expose OpenVDB
as the product capability or import this implementation facade directly.

Driver calls execute in the importing CPython process through the upstream
nanobind extension. The package does not launch a subprocess, daemon, RPC
service, or file-based native bridge. Backend entry-point discovery loads only
the lightweight descriptor; the numerical and native facade is imported when
the backend is inspected or executed. Numerical operations require a compatible
NumPy supplied and owned by the application rather than by this driver package.
Applications using the semantic API must let `sdf_tools` authenticate the
driver before importing this implementation package; direct driver preloads
are rejected by production discovery. The authenticated source loader compiles
approved driver bytes itself and ignores benign driver `__pycache__` content.
Low-level maintenance diagnostics should run in a separate interpreter so
their direct imports do not cross the production registry's preload boundary.

The native `openvdb==13.0.0+wu.3` wheel is built separately from the pinned
OpenVDB `v13.0.0` source commit. The driver verifies `LIBRARY_VERSION`, reports
the loaded extension path and SHA-256 digest, and maps available native
operations to the generic SDF capability vocabulary. Production startup also
requires an independently reviewed `promoted` release-lock record for the host
architecture and hashes every installed native wheel member against it.
Before importing the upstream wrapper or nanobind extension, the loader binds
the driver manifest to an identity embedded in authenticated driver source,
binds the release-lock digest from that manifest, requires a promoted
architecture record, verifies the exact wrapper, source lock, provenance,
distribution metadata, and native closure, and rejects executable additions,
symlinks, namespace/zip imports, and alternate package roots. Regular
installer-created `.pyc` and `.pyo` files are inert because neither wrapper nor
native code is loaded through bytecode resolution; they are ignored and are
not copied into the authenticated snapshot.
It copies the authenticated runtime bytes into a private read-only snapshot,
retains the snapshot root descriptor, and loads the extension and its
`$ORIGIN` dependency closure through that descriptor instead of reopening the
mutable installed names. It executes the captured wrapper source directly and
caches admission only after the loaded snapshot path and bytes have been
checked again. Semantic operations reuse that immutable admission with
fixed-size descriptor and inode checks; they do not rehash the native closure.
`inspect_runtime()` is the explicit diagnostic boundary that deeply rehashes
both the snapshot and installed promoted members.

This is an installed-distribution integrity boundary, not a defense against an
already compromised interpreter or process. Python startup code (`.pth`,
`sitecustomize`), hostile import hooks already present in `sys.meta_path`, and
process-level injection such as `LD_PRELOAD` must be controlled before the
driver is imported. Application packaging and deployment own that initialized
interpreter boundary.

```python
import numpy as np

import sdf_tools

mesh = sdf_tools.Mesh(
    vertices=np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],
        dtype=np.float32,
    ),
    triangles=np.array(
        [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
        dtype=np.int32,
    ),
)

session = sdf_tools.create_session(
    backend="openvdb",
    require={
        sdf_tools.Operation.MESH_TO_SDF,
        sdf_tools.Operation.OFFSET,
        sdf_tools.Operation.SAMPLE_VALUES,
        sdf_tools.Operation.FIELD_TO_MESH,
        sdf_tools.Operation.WRITE_FIELDS,
    },
    require_formats={sdf_tools.Operation.WRITE_FIELDS: {"vdb"}},
)
field = session.mesh_to_sdf(mesh, voxel_size=0.02, name="tetrahedron")
expanded = session.offset(field, 0.01)
sample_points = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
distances = session.sample_values(expanded, sample_points)
result = session.field_to_mesh(expanded, adaptivity=0.05)
session.write_fields("tetrahedron.vdb", expanded, format="vdb")
backend_evidence = session.backend_info.as_dict()
```

The content-agent driver intentionally advertises `WRITE_FIELDS` but not
`READ_FIELDS`. A session that requires `READ_FIELDS` fails capability admission.
The lower-level `openvdb_runtime.read_grid()` and `read_all()` functions exist
only for application code handling authenticated artifacts and require the
caller to pass `trusted_artifact=True`. Their regular-file, byte-size, metadata,
and post-read checks do not sandbox allocations made while native OpenVDB parses
a file; they must not be used for arbitrary model- or user-supplied VDB input.

Internally, `mesh_to_level_set` and `volume_to_mesh` prefer the repository's
`openvdb.tools` implementations, which expose deterministic `thread_count`
control and accept the caller's mesh and active-voxel ceilings before native
output allocation. A stock OpenVDB 13 module can use the official
`FloatGrid.createLevelSetFromPolygons` and `FloatGrid.convertToPolygons`
fallbacks. The repository extension adds copy-producing CSG, offset,
normalization, rebuild, transform-matched resampling, unsigned distance fields,
world-space sampling, topology conversion, and enclosed-region extraction.
`mean_filter` remains strict to level sets; use `scalar_mean_filter` for unsigned
distance fields and other scalar `FloatGrid` classes.

All resampling and sampling calls select `nearest`, `linear`, or `quadratic`
interpolation explicitly. Sampled gradients are expressed in world coordinates
per world unit. Rebuild and distance-field band widths are measured in voxels;
topology morphology steps and band widths are caller-visible and bounded, and
no repair or closing policy is applied implicitly.

The generic session exposes `backend_info` for workflow evidence. Driver and
native-package maintenance code can use `inspect_runtime()`, `is_available()`,
or `require_runtime()` to diagnose this implementation. `ExecutionLimits`
bounds mesh sizes, voxel extents, sample batches, morphology steps, thread
counts, band widths, aggregate filter work, returned active voxel counts,
field-memory totals, serialized metadata, and VDB file bytes. Writes retain and
verify a secure staging inode inside a private destination-local directory, then
atomically replace the destination only after the byte limit is verified.
The native module also enforces fixed active-voxel, grid-memory, coordinate,
and operation-work ceilings; application policy can provide stricter values.
These limits do not turn native VDB file parsing into an untrusted-input
allocation sandbox.

Installed backend discovery admits only the exact distribution and entry point
named by checked-in `sdf_tools` policy. The driver license manifest and native
closure gate must reject any LGPL component introduced or bundled by the SDF
delta; top-level OpenVDB licensing alone is not sufficient evidence. nanobind
and robin-map are compiled/bundled inputs and are included in that closure.
Application-owned NumPy and the external glibc platform ABI are recorded outside
the introduced/bundled claim and remain the application's dependency-review
responsibility. Candidate release records are review material only and are
rejected by both native build admission and runtime driver inspection.
