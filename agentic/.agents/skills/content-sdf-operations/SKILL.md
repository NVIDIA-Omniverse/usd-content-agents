---
name: content-sdf-operations
description: Use when a content agent needs bounded signed- or unsigned-distance-field operations such as mesh rasterization, field booleans, offsets, filtering, sampling, topology conversion, surface extraction, or admitted field output through an in-process backend.
metadata:
  author: NVIDIA Omniverse
---

# Content SDF Operations

Use the backend-neutral `sdf_tools` contract for signed- and unsigned-distance
field work. Select semantic capabilities first; treat the implementation that
provides them as recorded execution evidence, not as the capability surface.

## When To Use

- Convert a world-space polygon mesh to a signed or unsigned distance field.
- Apply bounded field booleans, offsets, smoothing, normalization, or rebuilds.
- Sample field values or gradients at declared world-space points.
- Convert active topology to an SDF, extract an enclosed region, or polygonize
  an isosurface.
- Write a backend-supported field artifact with an explicit format.
- Inspect which admitted backend and exact implementation executed an operation.

Use a qualified Geometry Repair workflow instead when the goal is to accept or
publish repaired geometry. A successful SDF operation does not establish source
fidelity, protected-feature preservation, correspondence, or repair acceptance.

## Ownership Boundary

This is an atomic capability, not a workflow. It owns one admitted in-process
SDF session and the bounded operations requested from that session. It does not
own run state, artifact lifecycle, refinement policy, geometry acceptance, or a
terminal workflow outcome. The calling workflow owns those decisions and must
bind this capability's execution evidence into its own durable handoff.

## Core Contract

```python
import numpy as np
import sdf_tools

mesh = sdf_tools.Mesh(
    vertices=np.asarray(vertices, dtype=np.float32),
    triangles=np.asarray(triangles, dtype=np.int32),
)
session = sdf_tools.create_session(
    backend="auto",
    require={
        sdf_tools.Operation.MESH_TO_SDF,
        sdf_tools.Operation.SAMPLE_VALUES,
        sdf_tools.Operation.FIELD_TO_MESH,
    },
)
field = session.mesh_to_sdf(mesh, voxel_size=0.01, half_width=3.0)
values = session.sample_values(field, sample_points)
surface = session.field_to_mesh(field)
backend_evidence = session.backend_info.as_dict()
```

`backend="auto"` is for exploratory work. For a benchmark, qualified geometry
route, or durable artifact, select the exact backend admitted by the governing
policy and persist `backend_info`. The first production-qualified driver is
`openvdb`, backed by the source-locked OpenVDB 13 runtime; future admitted
drivers may implement the same operations.

## Rules

- Keep world units explicit. Signed fields use negative values inside.
- Declare voxel size, band width, interpolation, isovalue, thread count, and
  resource limits rather than relying on an implementation default.
- Treat `Field` as an opaque, backend-owned in-process value. Do not unwrap its
  payload, mix fields from different backends, or serialize a live field handle.
- Request every operation and artifact format needed when creating a session,
  using `require_formats` together with `READ_FIELDS` or `WRITE_FIELDS` so the
  direction is explicit. An unavailable capability must fail admission; it is
  not permission to substitute another algorithm or lower-level API silently.
- Do not import `openvdb` or `openvdb_runtime` from content-agent code.
- Do not install, import by module path, or register a model- or caller-supplied
  backend. Installed discovery admits only exact entries in checked-in policy.
- Driver calls execute inside the importing CPython process. Do not add an SDF
  helper process, daemon, RPC service, JSON transport, or file-based bridge.
- Do not admit an SDF dependency delta that introduces or bundles an LGPL
  component. Require the structured license manifest and native-closure gate;
  a top-level package license alone is insufficient.
- Preserve resource-limit, unavailable, invalid-geometry, and backend-operation
  failures as distinct outcomes.

## Evidence

For durable outputs, record the requested operations, selected backend ID,
`backend_info`, execution limits, units, voxel and band controls, input/output
digests, and any selection rejections. Record the explicit artifact format for
field output. Backend identity is provenance, not a replacement for geometric
validation.

## References

- [Capabilities and selection](references/capabilities-and-selection.md)
- [Backend admission](references/backend-admission.md)
