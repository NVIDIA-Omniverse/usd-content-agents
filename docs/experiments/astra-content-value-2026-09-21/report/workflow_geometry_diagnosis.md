# Read-only diagnosis of the native Geometry failures

The Content Agents drawer (01), refrigerator hinge (03), printer (09) and Dextra hand (10) runs stopped at Geometry and explicitly declared `claimed_accepted=false`. None completed Geometry → Joint → Physics → Validation as an accepted physical asset. The successful direct Astra drawer is a separate result; it does not establish successful execution of the native Content Agents chain.

This diagnosis was performed after all affected evaluator contracts were frozen. It reads completed submissions, durable native receipts and the pinned public repository. No author output, repository implementation, source asset or frozen evaluator was changed. It does not rerun the scored author arms.

## Confirmed repair defect

A read-only call to `geometry_repair.protected_features.detect_protected_feature_candidates()` on case01's preserved normalized USD reproduces the exact tuple reported by the failed native receipt:

```
protected_features.py:475, in detect_protected_feature_candidates
    source_vertex_by_position[
KeyError: (0.000528343915939331, -0.00020667807757854465, 0.0004099169969558716)
```

The public [diagnosis evidence receipt](workflow_geometry_diagnosis_evidence.json) records the native failed receipts and exception tuple; the full retained read-only traceback is not included in this bundle. In repository commit `a96faf9cb2f5c1f655fe0d60c0ccf57e3477b1aa`:

- `agentic/packages/geometry_repair/geometry_repair/mesh_io.py:1124–1130` groups coincident vertices and computes averaged positions with `np.add.at` followed by division.
- `agentic/packages/geometry_repair/geometry_repair/protected_features.py:450–458` builds a dictionary from original exact floating-point position tuples. Lines474–479 look up boundary-loop positions from the welded analysis mesh in that dictionary. An averaged float need not equal any original tuple, so the lookup raises `KeyError` before a repair certificate is produced.
- `agentic/packages/content_agent_workflows/content_agent_workflows/geometry/workflow.py:3232–3270` catches the exception and exposes `str(exc)`, obscuring the exception type and stack as a tuple-valued error.

The smallest robust fix would preserve an explicit analysis-vertex → original-vertex index map through positional welding. It should not guess original identity with an unrestricted nearest-neighbor lookup. Regression tests should cover repeated positions after unit normalization, near-coincident seam vertices, stable source indexing, and unchanged original visual faces. A typed failure receipt should retain the exception type and a traceback artifact. These are proposed changes, not applied changes.

## Conversion boundary

Both runs invoked the installed `usd-convert-cad==0.2.0` command through the public conversion workflow. The converted glTF geometry retains numerical coordinates corresponding to meters while the USD stage declares `metersPerUnit=0.001`, yielding a 1000× physical scale discrepancy if consumed literally. Case03's numerical vertical extent is approximately2.18404, despite the millimeter metadata. Case01's source cabinet has18198 triangles; conversion retains18188. Its ten omitted source triangles are zero-area faces:41,44,47,51,54,57,92,95,102,105. The four drawer meshes retain2052 triangles each. Those facts are recorded in the arm's source/converter audit artifacts; they do not establish the proprietary converter's internal cause.

Relevant pinned code boundaries:

- `agentic/packages/content_agent_workflows/content_agent_workflows/convert_to_usd/workflow.py:77–78,802–827` routes glTF/GLB through CAD conversion and requests accurate tessellation, without a format-specific lossless visual-face or meter-unit contract.
- Installed `usd_convert_cad/cli.py` delegates to compiled HOOPS parameters. This diagnosis does not attribute the face deletion to a particular internal option.
- `agentic/packages/content_agent_workflows/content_agent_workflows/geometry/source_prep.py:835–855` permits a bounds-proven metadata-only coordinate correction for BREP inputs. That exception does not establish a valid glTF correction path.
- The 3MF cleaner/writer elsewhere in `source_prep.py` also removes degenerates and writes millimeter metadata, but it is **not the invoked glTF path** and is not evidence of the glTF defect's cause.

A bounded future fix should verify source-format units and transformed source bounds at conversion, preserve the original visual vertex/index arrays including collapsed source faces, and keep any collision cleanup in a separately identified collision representation. Metadata correction requires source/bounds evidence; changing stage units by habit is insufficient.

## Role-blind topology handoff

The first native Geometry receipts report670 cabinet boundary edges for case01 and6655 boundary edges for case03. These assets contain ordinary open visual surfaces. The native handoff applies one global mesh topology gate before a separate collision representation is authored:

- `geometry/workflow.py:503–675`, `_cad_preflight_checks`, always selects `mesh_topology` and calls `audit_usd_mesh_topology()` without a visual-versus-collision role scope. Failures enter the failed disposition.
- `agentic/packages/cad_verifier/cad_verifier/sim_ready.py:639–677` counts boundary/overconnected edges as non-manifold and fails meshes for non-manifold, degenerate, duplicate or non-triangle topology.
- `sim_ready.py:711–718` requires zero non-manifold edges and recommends watertight collision/render meshes for simulation handoff.

A future role-aware gate should retain faithful open visual geometry with explicit diagnostics, while requiring suitable independently validated collision geometry before physical acceptance. Globally disabling topology checks or filling a cabinet cavity would not solve the physical task. The repair package's visual-only profile cannot currently rescue the chain when its protected-feature detector crashes, and it does not remove the earlier role-blind handoff boundary by itself.

## Late printer and Dextra outcomes

The completed native Geometry receipts for both 09 and 10 record `success=false`, `validation_status=fail`, `handoff_ready=no`, and no exception in the initial run. Their failure is not the tuple-valued repair exception reproduced above. The retained native preflight reports these topology and triangle-budget observations:

| Case | Meshes | Degenerate faces | Duplicate faces | Boundary / over-connected edges | Triangle-equivalent faces |
| --- | ---: | ---: | ---: | ---: | ---: |
| 09 printer | 3,092 | 0 | 3 | 75,797 / 16 | 4,301,821 |
| 10 Dextra hand | 69 | 72 | 10 | 66 / 76 | 1,638,104 |

Both triangle-budget checks failed. These are observations on the imported visual geometry, before task collision bodies were authored; they do not establish that the original CAD design is physically defective. The [evidence receipt](workflow_geometry_diagnosis_evidence.json) binds the exact native Geometry result and validation-evidence hashes.

The printer preserved its first partial candidate, invoked one native visual-only repair, and interrupted its own repair process after the declared 900-second repair budget. Native source-format validation passed, but no repair certificate, repaired output or native terminal result was produced before interruption. The [budget-stop record excerpt](workflow_geometry_diagnosis_evidence.json) and [attempt closeout excerpt](workflow_geometry_diagnosis_evidence.json) are author records, not substitute native certificates. Joint preparation remained a draft; Joint and Physics authoring were not run. Standalone Validation rendered the partial asset but failed missing physics, and its assessment execution records `authorized=false`. Final scene and bindings remain byte-identical to the preserved first candidate.

Dextra invoked two repairs. The first was cancelled during intake after its default eight-attempt policy conflicted with the outer repair limit; no candidate was authored. The second used the public Geometry API with `max_attempts=1` and a 450-second repair budget. Its [native certificate excerpt](workflow_geometry_diagnosis_evidence.json) is `rejected`: no accepted attempt, no applied operations, no output hash, and unchanged render geometry. It records four blocker classes—degenerate faces, duplicate faces, inverted shells and over-connected edges—no approved automatic worker targeting them, and budget exhaustion. The retained [plan excerpt](workflow_geometry_diagnosis_evidence.json) contains only a no-op operation; the final result records zero attempts. Ranked worker names in routing metadata do not prove those workers ran or that an executable was missing. The enclosing Geometry wrapper was later interrupted at its 700-second outer bound, after the native rejection certificate existed. No replacement candidate was promoted; Joint and Physics were not run.

Dextra also exposes a separate expressibility limit in the pinned embedded Joint route. `articulation/embedded_decision.py:320–337` permits only six signed cardinal axis tokens; lines 3003–3045 map those tokens directly to `motion_axis_world`. The [retained CAD-axis derivation excerpt](workflow_geometry_diagnosis_evidence.json) gives the two thumb axes approximately `(0.173648, 0, -0.984808)`. Their independently recomputed angle to the nearest supported world axis is 10 degrees. This diagnosis verifies the public schema and that angular calculation; it does not rerun the original CAD-face extraction. The final submission discloses the limit and authors no approximate joint. Because Geometry already blocked the chain, this is an additional capability constraint, not an observed Joint execution failure.

The five linked excerpts above are in the published evidence JSON at zero-based records 19, 20, 24, 25 and 29 respectively. Each records the retained native or author file path and SHA256; the complete underlying files are retained privately and are not included in this public subset.

## Secondary environment limitation and scope

Case03 also records an `Ndr`/`pluginFactory` source-format validator initialization failure, leaving that check `not_evaluated`. It is a distinct infrastructure limitation, not evidence of incorrect source geometry. Any future capstone must qualify the pinned package combination before claiming a completed Validation stage.

The configured installation also lacked dependencies selected by other scored workflow routes. Case02 directly records unavailable Scene Optimizer build resources and the separately qualified Geogram/Vorpalite executable; those setup findings and the pinned public installation requirements remain in the [independent method review](independent_method_review.md) and [backend evidence](optional_backend_review_evidence.json). They must not be erased by treating all native failures as workflow defects, or imported into 09/10 as unobserved causes. Shared hardware and low-level tools do not establish complete workflow provisioning. Runtime contention, source representation and attempt budgets also limit causal interpretation. These results establish non-delivery in this configuration and do not establish a causal advantage for either authoring approach.

An isolated, explicitly unscored capstone can test three narrowly scoped changes: source-faithful glTF transfer/unit verification, stable protected-feature indexing with typed errors, and role-aware visual/collision handoff. It must then run the public Geometry → Joint → Physics → Validation chain and the independent five-seed drawer contact/load test. This diagnosis does not prove that completion or estimate a guaranteed repair time. Any later capstone is documented separately and does not replace these frozen scored outcomes; no patch or scored rerun was performed for this diagnosis.

Evidence roots: `runs/pilot-v1/{01_drawer,03_hinge,09_printer,10_complex}/content_agents/`. The parent report retains the frozen output manifests. Minimal receipt excerpts and their original hashes are recorded in `workflow_geometry_diagnosis_evidence.json` beside this report. Native workflow outcomes, independent physical acceptance and author-protocol eligibility remain separate.
