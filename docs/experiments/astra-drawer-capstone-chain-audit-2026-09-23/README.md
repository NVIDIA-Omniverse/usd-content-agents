# Drawer capstone: independent evidence audit

**The bounded physical demonstration is supported.** Native Joint03, Physics10 and Validation01 completed successfully, and the same final asset passed all five independent loaded-drawer trials. Geometry remains conditional; native automatic cross-stage integrity was not evaluated.

| Stage | Supported outcome |
|---|---|
| Geometry04 | [Conditional source-preserving handoff](https://github.com/NVIDIA-Omniverse/usd-content-agents/blob/864dfcdb91c46cf4def78cf9a292ca83b5f40aff/docs/experiments/astra-drawer-capstone-2026-09-21/runs/drawer_geometry_04/geometry_workflow_result.json); formal SimReady not evaluated |
| Joint03 | [Native completed / accept](https://github.com/NVIDIA-Omniverse/usd-content-agents/blob/864dfcdb91c46cf4def78cf9a292ca83b5f40aff/docs/experiments/astra-drawer-capstone-2026-09-21/runs/drawer_joint_03/standalone_articulation_terminal_receipt.json) |
| Physics10 | [Native pass](https://github.com/NVIDIA-Omniverse/usd-content-agents/blob/864dfcdb91c46cf4def78cf9a292ca83b5f40aff/docs/experiments/astra-drawer-capstone-2026-09-21/runs/drawer_physics_10/physics_behavior_assessment.json) for three seconds of constrained mounted rest |
| Validation01 | [Actual terminal pass / review accept](https://github.com/NVIDIA-Omniverse/usd-content-agents/blob/864dfcdb91c46cf4def78cf9a292ca83b5f40aff/docs/experiments/astra-drawer-capstone-2026-09-21/runs/drawer_validation_01_prepared/native_run/validation_terminal_receipt.json); four required gates pass without waivers |
| Independent physical task | [Five of five loaded opening/closing trials pass](https://github.com/NVIDIA-Omniverse/usd-content-agents/blob/864dfcdb91c46cf4def78cf9a292ca83b5f40aff/docs/experiments/astra-drawer-capstone-2026-09-21/evaluations/native10_source_clear_v1/report.json) on the same asset |

The final asset SHA256 is `0ef7038845d569a2f502f38af9fb5b1e4e99cf06c5f4e04035e4c259b1b07156`. The [final conjunction](https://github.com/NVIDIA-Omniverse/usd-content-agents/blob/864dfcdb91c46cf4def78cf9a292ca83b5f40aff/docs/experiments/astra-drawer-capstone-2026-09-21/evidence/final_capstone_conjunction.json) binds native Validation and task acceptance; the [independent trace audit](https://github.com/NVIDIA-Omniverse/usd-content-agents/blob/864dfcdb91c46cf4def78cf9a292ca83b5f40aff/docs/experiments/astra-drawer-capstone-2026-09-21/task_evidence_audit_v1/native10/audit.json) corroborates the recorded task metrics within stated limits. The [task plot](https://github.com/NVIDIA-Omniverse/usd-content-agents/blob/864dfcdb91c46cf4def78cf9a292ca83b5f40aff/docs/experiments/astra-drawer-capstone-2026-09-21/task_plots/native10_source_clear_v1/drawer_task.png) summarizes the five trials.

The successful chain includes [explicit preparation](https://github.com/NVIDIA-Omniverse/usd-content-agents/blob/864dfcdb91c46cf4def78cf9a292ca83b5f40aff/docs/experiments/astra-drawer-capstone-2026-09-21/runs/articulation_preparation_03/articulation_preparation_publication.json), packaging, upper-drawer rigid-body promotion and a [source-bound joint endpoint correction](https://github.com/NVIDIA-Omniverse/usd-content-agents/blob/864dfcdb91c46cf4def78cf9a292ca83b5f40aff/docs/experiments/astra-drawer-capstone-2026-09-21/evidence/static_endpoint_repair_protocol_v1.json). It is an extensively repaired, unscored follow-up. It does not establish an uninterrupted autonomous pipeline or controlled workflow advantage.

The task uses a free 0.5 kg payload, gravity and bounded external drawer force. Its ideal prismatic constraint excludes cabinet-to-upper-drawer contact. Estimated physical properties, fixed seeds, the [source-clear placement correction](https://github.com/NVIDIA-Omniverse/usd-content-agents/blob/864dfcdb91c46cf4def78cf9a292ca83b5f40aff/docs/experiments/astra-drawer-capstone-2026-09-21/SOURCE_CLEAR_FIXTURE.md), missing initial native pose components and missing per-contact actor IDs limit interpretation. Native mounted rest is distinct from loaded opening/closing. No hardware, rail-friction, actuator or universal SimReady claim follows.

## What can be verified publicly

The included helper rehashes all **544 published files**, reads existing stage verdicts and verifies five complete gzip traces with **2,460 samples each**. It does not run a model, renderer, solver, USD decoder or metric recomputation.

```sh
python verify_public_evidence.py --bundle PATH_TO_CAPSTONE_BUNDLE
```

Use the capstone directory from commit `864dfcdb91c46cf4def78cf9a292ca83b5f40aff`. The helper pins its publication-manifest hash. [Actual public-byte verification](public_verification.json) explicitly reports `full_original_terminal_closure_verified: false`: the accepted coordinator plan and plan patch are privacy projections. Their public bytes are verifiable; their unavailable original bytes cannot satisfy the original native terminal bindings. See [the original evidence notes](https://github.com/NVIDIA-Omniverse/usd-content-agents/blob/864dfcdb91c46cf4def78cf9a292ca83b5f40aff/docs/experiments/astra-drawer-capstone-2026-09-21/EVIDENCE_NOTES.md).

The separate retained-original audit passed 111 checks over 641 files. Its digests in [audit_summary.json](audit_summary.json) are **attestations**, not a claim that those private originals are supplied here. The public helper makes the narrower verification boundary explicit.

All evidence links are pinned to the existing publication commit and were checked against that commit's local Git objects. This audit does not change historical verdicts or create new acceptance. [Derivative manifest](manifest.json).
