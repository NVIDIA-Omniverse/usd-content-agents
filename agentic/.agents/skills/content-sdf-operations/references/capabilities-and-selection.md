# SDF Capabilities And Selection

## Semantic Operations

The public operation vocabulary is backend-neutral:

- mesh to signed or unsigned distance field;
- field to polygon mesh;
- union, intersection, and difference of signed fields;
- signed offset, smoothing, normalization, and rebuild;
- transform-matched resampling;
- value and world-gradient sampling;
- active-value masks, topology-to-SDF, and enclosed-region extraction;
- explicit-format field read and write, when a driver truthfully and safely
  declares each direction.

Create one `SdfSession` with all required `sdf_tools.Operation` values before
starting related work. The session binds one backend, one identity record, and
one `ExecutionLimits` value. Later calls do not switch implementations.
Pass every required artifact format through `require_formats` together with its
`READ_FIELDS` or `WRITE_FIELDS` operation. A flat set applies to each requested
I/O direction; use an operation-keyed mapping when the read and write formats
differ. Directional format support is part of capability selection and is
checked again before each read or write.

## Selection

Use `backend="auto"` only when exploring and when any admitted implementation
with the requested capabilities is acceptable. Persist both the selected
`backend_info` and `selection_rejections`.

Use an explicit backend ID for reproducible evaluation, qualified Geometry
Repair, or an artifact whose behavior was validated against one implementation.
If the backend is absent, its identity differs from policy, or a capability is
missing, stop with the typed unavailable result. Do not retry a different
backend implicitly.

A future NanoVDB driver can register the sampling or transport operations it
actually supports without claiming another backend's mutation and meshing
operations. Capability selection must reject the driver when the requested
operation set is larger than its truthful implementation surface.

## Field Ownership

Every `Field` carries a backend ID and semantic kind. A session refuses a field
from another backend before native execution. Cross-backend conversion requires
an explicit portable representation and a separately validated operation; it
must not be implemented by extracting a private payload.

Signed-field booleans and offsets require negative-inside signed-distance
fields. Unsigned fields and masks remain distinct even if a particular driver
uses the same native scalar storage for them.
