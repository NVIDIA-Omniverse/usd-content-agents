# Routing And Fidelity

## Source Decision Table

| Source | Default path | Fidelity claim |
| --- | --- | --- |
| `geometry.source.v1` | Verify manifest, choose an exported representation by role, preserve provider metadata | Provider-declared fidelity plus independently verified artifact identity |
| Existing readable USD | Preserve directly | `direct_usd_preserved` |
| STEP, IGES, BREP, URDF, MJCF, GLB, FBX, generic 3MF, supported mesh/CAD | Shared convert-to-USD workflow | `shared_conversion_noneditable` |
| STL/OBJ requested as editable | Explicit `parametric_recovery` with lossy-recovery consent | `parametric_recovery_request` |
| STEP/STL/OBJ compatibility input | Explicit `opaque_import` | `opaque_cad_import` |
| Prompt/image without exported artifact | Invoke an explicitly selected authoring provider first | Unsupported until an immutable source bundle exists |
| Unknown format | Preserve as an unsupported external-source blocker | `unsupported_requires_converter` |

`auto` must not silently choose lossy recovery, opaque import, provider-native
code execution, or provider-specific semantics. An explicit authoring mode must
match the source type. `.py` and other executable text never qualify as
geometry artifacts.

Within a source bundle, prefer `render_geometry`, then `design_exchange`,
`collision_candidate`, and `reference`, unless the caller requests one exact
role. Retain `native_source` only as inert provenance.

## Fidelity Rules

- Preserve the original source bundle and selected representation identities.
- Verify byte size and SHA-256 before conversion or validation.
- Preserve producer, revision, request digest, parent bundle, coordinates,
  parts, parameters, rights, and upstream edit URI.
- Record whether geometry remains editable in its upstream provider; an export
  does not recover provider feature history.
- Record every lossy, opaque, or reconstructed transition.
- Retain converter probe, report, validation, and manifest artifacts.
- Use content-aware USD detection rather than suffix-only acceptance.
- Do not infer B-rep from a filename or tessellated USD. Set `brep_source` only
  when an external artifact and correspondence evidence prove exact B-rep.
