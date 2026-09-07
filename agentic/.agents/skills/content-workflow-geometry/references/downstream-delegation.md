# Downstream Delegation

## Ownership

Geometry owns source-bundle identity, source fidelity, geometry intent,
semantic parts and parameters supplied by the provider, correspondence policy,
geometry readiness gates, and handoff metadata.

Delegate:

- visual and physical material assignment to `content-workflow-material`;
- colliders, mass, density, friction, restitution, and behavioral simulation to
  `content-workflow-physics`;
- joints and articulation inference to `content-workflow-articulation`;
- authored-physics or temporary-proxy runtime checks to
  `content-workflow-runtime-validation`;
- profile selection, conformance routing, and formal status to
  `content-workflow-simready`.

## Geometry Hints

Emit a hint only from explicit source metadata, provider evidence, or user
intent. Do not invent SDF settings, density, friction, restitution, insertion
axes, or physical materials. Hints are advisory, not authored physics, and
must identify their source and confidence when available.

## Runtime Modes

- `skip`: normal geometry-only handoff.
- `authored_physics`: validate an input that already contains rigid-body
  authoring; fail rather than invent missing physics.
- `temporary_loadability_proxy`: create disposable neutral proxy physics in
  the shared runtime workflow. Never feed it forward as final physics USD.
