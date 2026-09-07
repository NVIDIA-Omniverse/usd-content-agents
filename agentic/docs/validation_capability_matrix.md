# Agentic Validation Capability Matrix

The outer Codex or Claude reasoner supervises one decision-only planning child.
Deterministic preparation selects nothing; the child proposes a finite exact
check graph, and trusted outer code validates the digest-bound proposal before
any operation. Focused/direct compatibility still permits explicit outer
selection, but its classic plan is never agentic decision authority or fallback.

| Capability | Role | Required preflight | Authority and outcome |
|---|---|---|---|
| Coordinator preparation | Deterministic selection-free boundary | Exact source dependency closure, task/request/config identity, approved adapter inventory, evidence inventory, and mandatory constraints. Bare unresolved `OmniPBR.mdl` is bound as the canonical external renderer-runtime module; other unresolved or package-relative MDL dependencies fail closed. | Invokes no provider or operation, injects no dependency bytes, and publishes no classic plan. |
| Child launch custody | Parent-owned deterministic boundary | Complete runner/model/mode identity, staged skill policy, and exact child output paths | Freezes the descriptor before launch; acceptance rejects any child rewrite by digest. |
| `ValidationCoordinatorPlanPatch` | Decision-only Codex or Claude child proposal | Exact preparation digest and public patch schema | Proposes claims/criteria and one instance per selected capability. It cannot execute, assess, review, publish, or mutate source. |
| Accepted-plan receipt | Trusted outer validation | Known unique capability/template/rule IDs, exact targets/focus/typed parameters, declared dependencies, acyclic graph, and current preparation/policy | Authorizes exact adapter calls and only then emits the classic request/plan compatibility projection. |
| `physics_sane` / `physics.usd_schema_sanity` | Deterministic static check | Local OpenUSD schema runtime | Produces factual static findings on the standard Validation result contract. |
| `render_valid` / `render.runtime_evidence` | Renderer or current render-evidence leaf | Final geometry evidence from the shared OVRTX USD render path, retaining the exact source USD digest and OVRTX render metadata under the [canonical visual-evidence contract](../packages/content_agent_workflows/docs/verified_operation_ingress.md#canonical-post-mutation-visual-evidence) | Binds render paths/digests and render validity; a precomputed OVRTX image needs an exact `qualified_render_evidence` source/image/backend/metadata binding, the v1 qualified path rejects multi-source requests, generic/local previews remain reference-only, and deterministic code does not decide semantic visual quality. |
| `physical_behavior` / `physics.behavior_evidence` | Runtime/simulator evidence consumer | Existing simulation, runtime, trajectory, video, or recording evidence | Produces an independent runtime verdict. It does not start Newton or another simulator. |
| `look_right` / `visual.optional_advisory_critique` | Optional external critique leaf | Passed current-run `render_valid` result and an explicitly configured judge provider | Produces provenance-distinct advice only. Missing or unrequested critique is never canonical authority. |
| Outer image and fact assessment | Outer-reasoned semantic step | Digest-bound render/reference images and required typed evidence | The one outer reasoner authors independent static, runtime, visual, package, and cross-stage dispositions. |
| Package finalizer | Deterministic identity/readback check | Completed mandatory selected operations and unchanged source closure | Publishes the #38-compatible result, explicit `not_requested`/`not_evaluated` states, and package evidence. |
| Cross-stage evidence binding | Deterministic composed-mode check | Accepted Articulation, Material, Texture, and Physics handoffs plus cross-stage receipt | Binds upstream evidence before assessment. It is `not_evaluated` in standalone mode, not a pass. |
| Assessment/review finalizer | Deterministic receipt validator | Outer-authored assessment and exact saved-output review | Publishes the same `validation_terminal_receipt.json` shape in standalone and composed modes. |
| `ingest-verified-operation-result` | Provider-free byte/provenance verifier | Domain-owned v1 envelope, projection, native report/payload, and exact artifact/component bindings | Executes nothing. Preserves each native operation/status independently for outer exact-digest assessment and terminal carry-through. |
| `produce-canonical-visual-evidence` | Package-owned usd-cli/OVRTX evidence producer | Exact post-mutation USD, complete dependency closure, package-owned usd-cli provenance, and explicit local or remote OVRTX backend | Produces digest-bound images, render responses, camera records, usd-cli journal/checkpoint/source revision, and render metadata, not a semantic verdict. The outer reasoner remains visual authority; ingestion is a separate call. |

Execute mode and provided mode are mutually exclusive. Provided mode cannot
fall back to templates and cannot rerender, simulate, normalize a domain
report, or call a provider. Its terminal receipt keeps every imported native
operation separate; the existing five broad gates remain explicit
`not_evaluated` rather than being inferred from domain-specific evidence.

`validate run` is the default agentic coordinator surface, but it has no
provider or rendering-backend default: callers select the planning runner and
any requested rendering backend explicitly. `validate run
--direct-executor` and `validate resume` retain the ordered-template
compatibility workflow. Focused execute mode uses `prepare` through
`review-assessment`, and provided mode starts with
`ingest-verified-operation-result`; every mode requires a separate run
directory. The config/Python `validation-agent` fixed pipeline is a separate
interface and retains its existing contracts.

Agentic coordinator runs never enter that compatibility resume path. A resume
request against one writes `validation_safe_restart.json`, reports
`safe_restart_required`, preserves every prior artifact, and invokes no legacy
executor or nested agent. The retry uses a fresh coordinator run identity and
empty output directory.

V0.6 permits one instance per capability and no revision, repeated-check,
carry-forward, or follow-up loop. The terminal coordinator chain binds
preparation, child proposal, accepted plan, exact operation results/index,
independent assessment/review, `publication_kind: validation_assessment`, and
`source_mutated: false`.

## Named Profiles

Profiles are stable request shorthands selected by the outer reasoner:

| Profile | Deterministic expansion |
|---|---|
| `static` | `physics_sane` |
| `visual` | `render_valid` |
| `runtime` | `physical_behavior` |
| `comprehensive` | `physics_sane`, `render_valid`, `physical_behavior` |
| `comprehensive-with-advisory-critique` | `physics_sane`, `render_valid`, `physical_behavior`, `look_right` |

Unselected capabilities remain `not_requested`. A selected optional advisory
critique that is not run remains `not_evaluated`. Missing mandatory selected
evidence fails closed. Selecting advisory critique also requires explicitly
selecting `render_valid` earlier in the same frozen request.

See
[`content-workflow-validation/SKILL.md`](../.agents/skills/content-workflow-validation/SKILL.md)
for the complete method, including explicit fail-closed invalidation of legacy
receipt/checkpoint identities that lack the current bindings.
