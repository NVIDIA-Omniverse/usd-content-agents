---
name: geometry-source-intake
description: Classify and prepare one geometry.source.v1 bundle or CAD, mesh, USD, URDF, MJCF, GLTF, GLB, or 3MF artifact while preserving units, provenance, editability, representation roles, and source fidelity. Use before Geometry, repair, segmentation, material, physics, or SimReady workflows need canonical geometry input.
metadata:
  author: NVIDIA Omniverse
---

# Geometry Source Intake

Produce one typed routing and preparation decision. The calling workflow owns
run state, evidence acceptance, artifact lifecycle, and completion.
Ownership is atomic: this skill does not own run state, evidence acceptance,
artifact lifecycle, or completion.

## Procedure

1. Freeze the source path or URI, manifest and artifact digests, format,
   caller-provided units and axes, rights assertion, provenance, references,
   requested representation role, and mutation authorization.
2. When `geometry.source.v1` is supplied, validate it before reading artifacts:
   - require a regular, bounded manifest;
   - resolve representation paths beneath the manifest directory;
   - verify each selected artifact's size and SHA-256;
   - prefer `render_geometry`, then `design_exchange`,
     `collision_candidate`, and `reference` unless a role is explicit;
   - retain producer, revision, coordinates, parts, parameters, rights, and
     provenance as metadata;
   - never execute `native_source` or any provider-authored code.
3. Otherwise classify with
   `content_agent_workflows.geometry.route_geometry_request` and prepare with
   `prepare_geometry_source` or the equivalent service operation. Do not
   reproduce converter or parser logic in a prompt.
4. Select the least lossy admitted route: direct USD preservation, shared
   conversion, explicit opaque import, or caller-approved parametric recovery.
   Generation and revision happen before intake through an explicitly selected
   `GeometryAuthoringProvider`.
5. Preserve animation, skin, morph, assembly, material, subset, primvar,
   articulation, and external-dependency semantics. Block a static fallback
   when it would erase required behavior or identity.
6. Never guess physical scale for raw polygon formats. Require authoritative
   units and up-axis before physical, repair, collision, or simulation claims.
7. Reject `.py`, code blocks, commands, and executable text. An external
   provider must export a supported immutable artifact first.
8. If geometry is defective, emit diagnosis and route the immutable source to
   `content-workflow-geometry-repair`. Do not repair during intake.
9. Return the selected route, prepared artifact and digest, fidelity and
   editability claims, representation role, source-bundle metadata, reports,
   warnings, and unresolved dependencies.

## Required Distinctions

- A provider-native source representation is provenance, not workflow code.
- `design_exchange` may preserve exact design geometry but does not imply
  editable provider history.
- Shared conversion is usable geometry, not recovered feature history.
- A tessellated USD is not exact B-rep and must not populate `brep_usd`.
- A normalized copy is not optimized geometry.
- A discovered feature is evidence to protect, not authority to invent or
  mutate geometry.
