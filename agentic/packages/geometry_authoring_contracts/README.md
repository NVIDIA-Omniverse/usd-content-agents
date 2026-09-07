# Geometry Authoring Contracts

Lightweight, provider-neutral contracts for external geometry-authoring
systems. This package contains typed requests, capabilities, receipts,
failures, immutable artifact bindings, and the `geometry.source.v1` handoff.
The handoff can retain non-runnable `supporting_asset` representations, such as
an OBJ material library or an external glTF buffer, beside the root geometry.
Supporting assets are digest-bound and staged with the bundle, but are never
selected as root geometry.

Semantic parameters carry scalar type, current value, units, optional bounds,
step, choices, UI grouping, semantic role, affected parts, and declared effect
domains. `GeometryAuthoringFamilyRequest` validates named variant rows against
those definitions before any provider invocation.

It does not contain a geometry kernel, execute provider-native source, or
select an authoring provider implicitly. Providers run behind independently
configured artifact-JSON or HTTPS adapters and must bind every response to the
exact request and artifact digests.

```python
from geometry_authoring_contracts import (
    GeometryAuthoringProvider,
    GeometryAuthoringRequest,
    GeometrySourceBundle,
)
```

Reference integrations live in `geometry-authoring-connectors`. The Geometry
Agent service registers those integrations explicitly and hands accepted
source bundles to `content-agent-workflows`.
