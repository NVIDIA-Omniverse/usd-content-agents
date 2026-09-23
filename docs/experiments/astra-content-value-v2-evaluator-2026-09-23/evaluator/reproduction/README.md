# Reproduce the cases02–10 evaluation

This supplement was prepared after the v2 evaluator freeze for publication after authoring ends. It is not an author input and changes no scoring code, reference, task or threshold. Its separate manifest binds the copied historical source builders and acquisition catalog.

## Replay versus regenerate

Replay uses `../cases/<case>/` including its complete `reference/` directory, the original source closure, and a separately retained submitted USD/bindings/dependency closure. `../freeze_manifest.json` binds 4,072 code/reference/task files, including 3,883 reference instances. `../qualification_manifest.json` binds the preparation evidence. Verify both manifests before evaluating. Publishing only the author-facing inventories is insufficient: they omit reference NPZ filenames and digests. A public replay bundle must supply all frozen reference files, or explicitly require their reconstruction or separate download.

`../invocations.json` gives structured CLI arguments for every case, report paths and verdict mappings. Relocate its installation prefixes and submitted/output paths; pass the original case asset ancestor as `--source-root`, not the narrower authoring subdirectory. Always override `--solver-python`: the historical CLI default points to the v1 installation. Run from a trusted evaluator directory with authored `PYTHONPATH` removed. Never execute submitted scripts or use submitted pass claims/trajectories as evidence. The orchestrator must independently contain restored USD dependencies and enforce resources.

Example, on a qualified Linux installation:

```sh
"$STRUCT_PY" "$EVAL/cases/09_printer/evaluate.py" \
  --case 09_printer --usd "$SUBMISSION/final.usd" \
  --bindings "$SUBMISSION/bindings.json" \
  --inventory "$EVAL/cases/09_printer/reference/source_inventory.json" \
  --source-root "$ORIGINAL_ASSETS/09_printer" \
  --solver-python "$SOLVER_PY" --device cpu --output "$NEW_RESULT"
```

The output directory must be new. Read `acceptance.json`; process exit zero alone is not acceptance. The normal invocation runs seeds11,23,47,83,131, plus case06's paired unloaded seed11. `--structural-only` is diagnostic and cannot establish functional acceptance.

Use the published v2 environment pins/constraints and runtime parity evidence. The source reference builders recorded NumPy2.5.3, Trimesh4.12.2 and cadquery-ocp7.8.1.1; STEP/FCStd also need its VTK9.3.1 dependency. The structural process uses OpenUSD25.5, while `ovphysx==0.4.13` runs in a separate Python3.12 environment with NumPy2.4.4. Do not import the structural `pxr` provider into the native solver process. Qualification used L40 hosts with the CPU solver backend; compatibility with a GPU-free host was not established. No fresh package installation is claimed by these instructions.

Scored independent evaluation reserves four CPUs and32GiB, with initial four-CPU affinity and one numerical-library thread. Complete structural/source inspection is bounded at1800s; input USD loading is separately900s; native trials remain900s except conveyor300s. The original2CPU/8GiB source stress timed out at900s and is retained. The actual-budget full3092-part retessellation check passed1266.87s and reached11.27GiB child RSS; it does not establish arbitrary collider-cooking cost. A timeout or failed runtime calibration is inconclusive, while independent concrete source/schema/physical failures remain failures as declared in each task.

## Obtain and verify originals

`source_catalog.json` records each public repository URL, exact commit/download identity, primary input, every selected original file SHA256, source-closure hash, license statement and unresolved licensing caveat. `original_source_dataset.json` is the exact source dataset used by the original independent freezer (SHA256 `34c8293ab1be49cd0bbdb8f8fe25278b60ba21c35d52b8009a5065278d25c87f`). Fetch the pinned commits/files and reconstruct the recorded relative directory layout. Verify all file hashes before conversion; repository names or matching filenames alone are insufficient.

Retain source license/attribution notices. In particular, the vise has a README MIT declaration with missing linked license text; the engine and Thor FCStd document metadata disagree with their repository grants; Voron publishes GPLv3 text without an independently established only-versus-or-later qualifier. These are recorded source facts, not newly resolved permissions. No alternative geometry may replace an unavailable source.

## How the references were made

|Case|Frozen instances|Generation and scope|
|---|---:|---|
|02 conveyor|4|Placed STEP solids/shell components; rail, carrier, support and payload names remain in source IDs.|
|03 refrigerator|8|glTF scene node instances in metres, using source node transforms; split primitives are separate instances.|
|04 gripper|27|Placed STEP solids; grounded jaw/arm/cam roles and grasp site are separately fixed in `case04_contract.json`.|
|05 vise|13|STEP assembly occurrence placements, including repeated parts and external STEP dependencies.|
|06 engine|24|Saved FreeCAD ZIP/XML/BREP terminal shape instances. Part-local shape/instance coverage is retained; original Assembly4 mate evaluation is not certified.|
|07 Thor arm|566|Saved linked FreeCAD terminal shapes and arrays, without FreeCAD execution/recomputation. Exact original mate frames and differential motor transmission are not certified.|
|08 excavator|80|All mesh instances of the independently saved `usd-convert-cad==0.2.0` conversion. Completeness relative to native SolidWorks and mate fidelity are unproven.|
|09 printer|3,092|All placed STEP solids of the original master assembly; principal axis/bed roles selected from recorded source names/solid indices.|
|10 Dextra hand|69|Every BREP shell in nine saved top-level cached components, with summed child face coverage checked. Four fingers have three uniquely matched phalanges each; thumb has two. No tendon/hardware accuracy claim.|

The unchanged general builders are in `builders/general/`. `geometry.py` uses OCP tessellation with0.025mm chord and0.1rad angular tolerances, a millimetre-to-metre scale for CAD, and scale1 for glTF. FCStd loading reads saved ZIP/XML/BREP content; it neither executes macros nor recomputes the original assembly. Non-surface construction objects remain in original files but are omitted from the surface inventory. STEP occurrence transforms are composed; FCStd part-local-only records are explicitly labelled.

For cases02–07 and09, a fresh geometry-only rebuild can be requested from a frozen case's `geometry.py` using the catalog's primary path and unit scale. This retains the actual importer implementation; the v2 comparison helper is not involved in conversion:

```sh
"$STRUCT_PY" "$EVAL/cases/05_vise/geometry.py" \
  --asset-root "$ORIGINAL_ASSETS/05_vise" \
  --source 'source/Machine Vice  Assembly.STEP' --unit 0.001 \
  --case 05_vise --output "$NEW_REBUILD/05_raw"
```

The complete metadata/role layer was produced by the retained `freeze_dataset.py` and `finalize_roles.py` against the exact dataset. Run copies in a new writable rebuild tree, never inside a frozen evaluator. Root arguments and metadata paths are installation-specific; changed absolute roots can change inventory JSON hashes without changing geometry bytes.

Cases03 and05 have a retained independent fresh regeneration qualification: all13 vise NPZ files match exactly; all8 refrigerator NPZ files match as a complete SHA multiset. Trimesh assigns random suffixes to three refrigerator split-primitive frame names, so direct filename ordering differs. Run `builders/align_reference_rebuild_by_hash.py --inventory FROZEN_INVENTORY --rebuilt NEW_REBUILD --output NEW_ALIGNED` to copy only exact regenerated bytes into the expected filenames. It fails unless the full hash multisets match and does not need the original NPZ files. See `evidence/reference_regeneration_v1.md`. No equivalent fresh byte-identical regeneration is claimed for the other seven cases.

For08, `freeze_dataset.py` invoked `usd-convert-cad` with `--up-axis z --instancing-style none --composition-style none --no-dedup --convert-metadata`. The resulting neutral USD is retained here as `intermediates/08_excavator/reference.usdc`, SHA256 `526380cbae3b02a57713212a2114507a68a906e58df741bd8bc4ab1b5460f8c4`. `builders/case08_10/case_source_freeze.py` enumerated its80 visible meshes and assigned boom/stick/bucket/base roles from recorded parent display names. This intermediate supports inspection of the reference extraction without claiming a fresh converter run reproduces identical bytes or all original CAD.

For10, the same retained builder enumerated `TopAbs_SHELL`, preserved cached shape coordinates, checked that child face counts sum to the parent, and matched phalanx template BREP volume/area. It expects a prior nine-component inventory from `freeze_dataset.py`. That intermediate inventory was not found among the extracted retained files; the final inventory preserves its expected SHA256. Regenerating the nine-component stage is possible through the retained general builder but has not been freshly qualified. The historical08/10 helper also has an explicit installation-specific `ROOT` constant; copy it into a separate rebuild tree or set module `ROOT`/`HERE` before calling its functions. Do not present its unchanged direct CLI as installation-independent.

## Recorded choices and remaining limits

Reference geometry generation is deterministic only within the demonstrated version/platform scope. The task roles are evaluator choices grounded in named parts/geometry, not automatically inferred hardware truth. `finalize_roles.py` contains the explicit Thor name-suffix mapping and printer solid indices; the gripper contract contains its solid-role table and payload site. Conveyor roles are the original product names. Engine roles follow the recorded source object names. Excavator display-name and hand shell/template mappings are explicit in the retained08/10 builder. Original transforms, usable source bounds and these grounded choices were inspected before authoring.

The printer payload placement was independently rebuilt from its frozen original bed meshes and exactly matched the saved placement JSON; case04 source rays and cam sign were also rechecked. These checks do not make fresh regeneration of every CAD reference byte-qualified. All full frozen NPZ/inventory files remain the scoring reference; any replacement needs a new experiment version. Materials/textures, native assembly completeness, physical motor models and manufacturing accuracy are outside the stated geometry/task acceptance scope.
