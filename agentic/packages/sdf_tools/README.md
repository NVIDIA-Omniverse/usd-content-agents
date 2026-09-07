# SDF Tools

`sdf-tools` is the backend-neutral Python surface for bounded signed-distance
field work in content-agent and geometry workflows. Callers select semantic
operations such as mesh rasterization, field booleans, offsets, sampling, and
surface extraction. They do not import a native library binding.

Backends are separately packaged drivers admitted by the checked-in backend
policy. Production registries load drivers only through exact admitted installed
entry points. Before importing driver code, discovery authenticates the owning
distribution's exact version, complete Python package, and policy-relevant build
manifest; production registries do not expose direct factory registration. The
first production-qualified driver is the repository-built OpenVDB 13 runtime.
Driver calls execute in the importing Python process; this package does not
create a helper process, daemon, or transport protocol.

Production discovery binds the admitted package namespace to a locked source
loader. The loader re-reads and hashes each policy-approved source file before
compiling it, never reads or writes interpreter bytecode, and prevents an
earlier `sys.path` package from shadowing the admitted distribution. Benign
`__pycache__` content is therefore inert, while `.pth` files, native import
artifacts, modules preloaded outside this authenticated loader, and unrecorded
executable package members fail admission before driver code runs.

```python
import sdf_tools

mesh = sdf_tools.Mesh(
    vertices=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    triangles=((0, 1, 2),),
)
session = sdf_tools.create_session(
    backend="auto",
    require={
        sdf_tools.Operation.MESH_TO_SDF,
        sdf_tools.Operation.FIELD_TO_MESH,
    },
)
field = session.mesh_to_sdf(mesh, voxel_size=0.01, half_width=3.0)
surface = session.field_to_mesh(field)
```

`backend="auto"` considers only drivers marked production-qualified in the
checked-in policy and is appropriate when any admitted implementation of the
requested capabilities is acceptable. A qualified route must replace `auto`
with the exact backend ID whose behavior and provenance it validated, then
persist `session.backend_info` as execution evidence. Fields are opaque,
backend-owned values; mixed-backend operations are refused before native code
runs.

When a workflow will read or write a field artifact, include the corresponding
`READ_FIELDS` or `WRITE_FIELDS` operation and pass its required identifiers
through `require_formats={...}`. A flat set applies to every requested I/O
direction. Workflows that need different read and write formats can pass a
mapping keyed by those operations. Backend identity exposes immutable
`read_formats` and `write_formats`; `supported_formats` remains their union for
compatibility. Selection and individual calls reject a format unsupported in
the requested direction before driver code runs.

Each session carries explicit execution limits for mesh, field, sampling,
threading, file, and metadata resources. Drivers receive those limits on every
operation; write metadata is converted to bounded finite JSON-compatible data
before a driver is called.

Capabilities remain driver-specific. In particular, the OpenVDB driver supports
explicit VDB writes but does not advertise field reads because native VDB parsing
does not yet provide the immutable-handle and allocation guard required for
untrusted content-agent input. Requiring `READ_FIELDS` therefore fails backend
selection instead of silently exposing a lower-level parser.
