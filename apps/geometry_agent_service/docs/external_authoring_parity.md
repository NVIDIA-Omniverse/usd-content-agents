# External Authoring Parity

Geometry Agent owns provider selection, immutable handoff, validation, repair,
USD preparation, SimReady geometry checks, and OVRTX evidence. External
authoring providers own construction and mutation of their native models.

This split retains a rich parametric authoring experience without embedding a
provider kernel, provider-specific model representation, or execution runtime
in Geometry Agent.

## Portable Authoring Surface

| User need | Public behavior | Provider responsibility |
| --- | --- | --- |
| Text-to-geometry | Submit bounded text intent to one selected provider | Construct a native model and export requested geometry |
| Image-to-geometry | Bind digest-verified reference images to the request | Interpret images and construct the model |
| Editing | Revise one immutable source revision | Preserve native history and return a new revision |
| Semantic parameters | Carry typed values, units, bounds, steps, choices, groups, and declared effects | Define and apply meaningful model controls |
| Parameter families | Materialize named variants from the same immutable base in parallel | Apply each validated parameter row independently |
| Assemblies and parts | Preserve named part hierarchy and source transforms | Return semantic part bindings |
| Provider checks | Retain provider assertions as advisory evidence | Report bounded checks and metrics |
| Editable source | Retain native source as inert, digest-bound provenance | Return the native representation when authorized |
| Multi-format output | Request and verify exact formats | Export every requested representation |
| Simulation handoff | Run common Geometry workflow and OVRTX evidence | No provider-specific simulation code is required |

Kernel-specific feature trees, selectors, constraints, and modeling commands are
not translated into a lowest-common-denominator API. They remain in the native
source and may be edited only by the provider that produced that revision.

## Reference Provider Mapping

- **Build123d**: a separately deployed worker may expose text/image generation,
  native Python source, semantic parameters, revisions, named parts, and
  explicit re-export of an immutable revision.
- **ForgeCAD**: an operator-authorized worker uses the same protocol. Geometry
  Agent contains no ForgeCAD runtime, installer, or license grant.
- **Onshape**: the user's MCP client connects to the official Onshape Labs
  FeatureScript MCP for parametric authoring and testing. The user then exports
  a supported file manually or uses an optional local API-key helper that first
  freezes a mutable workspace as an immutable version. The helper returns the
  same bounded source bundle used by other handoffs. The deployed Geometry
  Agent service receives no Onshape credential or MCP session.

Delegated Build123d, ForgeCAD, custom workers, and the optional local Onshape
export helper return the same bounded `geometry.source.v1` bundle. A manual
Onshape export can instead use the ordinary existing-asset path. The Onshape MCP
itself does not claim export or bundle production. Geometry Agent never executes
returned Python, JavaScript, FeatureScript, or another provider-native
representation.

When a bundle contains multiple exported representations, Geometry Agent
selects an exact `design_exchange` before a `render_geometry` mesh. If several
representations have that role, the unique representation bound to a root
semantic part is preferred. Every other representation remains in the
immutable bundle for downstream use; the selected representation ID is passed
explicitly into the shared Geometry workflow so publication and preparation
cannot independently choose different artifacts.

## Family Semantics

A family request names one base source and up to 64 variants. Every variant:

1. starts from the same immutable base bundle;
2. uses only declared semantic parameters;
3. is checked against type, unit, bounds, step, and choices before invocation;
4. is sent to the same explicitly selected provider with no fallback;
5. must return lineage to the base and the requested parameter values; and
6. is published as an independent content-addressed source bundle.

The job fails when any variant fails, while retaining typed results for all
attempted variants. A failed variant is never represented as a successful source.

## Deployment Boundary

The public connector package includes a framework-neutral worker runner and a
fixed-command backend suitable for an independently isolated provider service.
The command is selected by the worker operator, never by an API request. The
backend uses no shell, writes a bounded request file, expects one typed result
manifest, enforces a timeout, and redacts provider process output from public
errors.

Provider packages, credentials, model services, and licenses belong only in that
external worker deployment. They are not dependencies of the Geometry Agent
service or its public image.
