# Texture Capability Matrix

The outer Codex or Claude reasoner chooses these skills and operations directly.
The table is a contract reference, not an executable selector.

| Skill | Operation | Provider required | Mutation | Required output |
|---|---|---:|---:|---|
| `content-texture-scope` | `texture agentic-leaf uv-prepare` | No | `inspect` never mutates; `generate_missing` may author only selected non-variant mesh parents with absent UVs through the bounded box fallback, including explicit subset-to-parent promotion | Typed UV result plus exact requested-scope, UV, purpose-specific material, frozen-closure, and saved-stage readback; when asset-graph selected, a non-executing verified-operation projection and receipt-path publication |
| `content-texture-scope` | `texture prepare` | No semantic provider | No | Frozen request, deterministic scope/UV/material/dependency facts, references, initial OVRTX evidence, explicit status table |
| `content-texture-scope` | `texture propose` | Explicit Texture proposal provider | No | Advisory provider proposal and provenance |
| `content-texture-candidate` | `texture generate` | Explicit named image/texture generator | Candidate only | Generator inputs/provenance, exact unit artifacts, candidate binding |
| `content-texture-candidate` | `texture apply-provided` | No; deterministic apply | Candidate only | Exact ordered outer-generated image bindings/provenance, per-unit apply receipts, candidate binding |
| `content-texture-quality` | `texture evidence` | OVRTX renderer, no semantic provider | No | Static checks and matched source/candidate images; semantic status `not_evaluated` |
| `content-texture-quality` | `texture critique` | Explicit domain VLM | No | Advisory findings and provider/model provenance |
| `content-texture-quality` | `texture review` | Outer multimodal reasoner | No | Exact per-unit accept/reject/revise decision over references and matched images |
| `content-texture-publish` | `texture publish` | No semantic provider | Canonical publication | Exact candidate copy, full deterministic recheck, saved-stage readback, receipt |

The provider-free UV leaf has no caller-selectable projection mode and does not
provide canonical projection or atlas preparation. Those full preparation
capabilities remain on the existing Scene Optimizer Texture prepare path. Its
source may be a `.usd`, `.usda`, or `.usdc` layer, or a self-contained `.usdz`
package. A package with external dependency bindings is rejected before
invocation. When `generate_missing` authors an overlay over a package, the
output closure retains the exact sealed USDZ predecessor path, digest, and size.

## Selection profiles

| Task profile | Inspect | Propose | Generate | Evidence | Critique | Review | Publish |
|---|---:|---:|---:|---:|---:|---:|---:|
| Scope diagnosis | requested | not_requested | not_requested | not_requested | not_requested | not_requested | not_requested |
| Generate and outer-review | requested | optional | requested | requested | optional | requested | requested |
| Comprehensive final validation/publication | requested | not_requested | already completed | requested | optional | requested | requested |

`optional` means the outer plan must resolve the leaf to `requested` or
`not_requested`. It never means implicit execution. `not_requested` and
`not_evaluated` never satisfy a mandatory safety, evidence, review, or
publication gate.

The provider-free Texture asset-leaf catalog exposes this explicit chain:

```text
texture.uv-prepare.v1
  -> texture.prepare.v1
  -> texture.apply-provided.v1
  -> texture.evidence.v1
  -> texture.review.v1
  -> texture.publish.v1
```

Each leaf executes only its named focused operation below a distinct active
attempt, emits a native saved-stage readback and terminal receipt, and is
projected by a non-executing Texture-owned verified-operation projector.
`texture.prepare.v1` must bind the exact passing UV result and consume that
result's output and dependency closure; catalog dependency alone is not
completion dataflow. Therefore this six-leaf chain selects `generate_missing`,
never an inspect invocation that could terminate `not_evaluated`. Inspect-only
UV graphs must have no selected dependent and need a fresh graph if later work
requires preparation.
Proposal, provider generation, and critique remain focused outer-selected
operations outside this provider-free graph chain.

For a generated candidate, the outer reasoner selects exactly one execution
mode: `texture generate` or `texture apply-provided`. The latter consumes
candidate images created by an outer-selected image-generation capability; it
does not reinterpret reference images or call the built-in generator again.

## Qualification boundary

This decomposition and closure of parent #1142 demonstrate architecture
readiness only. Keyboard #1080, Dishwasher #1092, and reference-faithful Texture
qualification #1098 each require a genuinely fresh run identity and fresh direct
evidence after the architecture lands. Earlier attempts cannot be reused,
relabeled, or promoted by this capability matrix.
