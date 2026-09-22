# Reproduce source references and acceptance runs

The portable inputs are the original source closure in `assets/`, `protocol/dataset.json`, this evaluator directory, and each immutable `frozen/<case>/` snapshot. No Horde service or internal asset database is required to read the saved references. Obtain originals from the exact public URLs/commits in each asset manifest and verify every dataset hash. Redistribution remains subject to the individual source licenses and recorded licensing uncertainties.

## Recorded environments

The pilot used Linux x86_64, Python3.12.13, and public Content Agents commit `a96faf9cb2f5c1f655fe0d60c0ccf57e3477b1aa`. Use that repository's documented environment setup. Its observed USD provider was `usd-exchange==2.3.0`, exposing OpenUSD25.5. Do not install a second `usd-core` provider into the same environment. The complete observed evaluator dependencies are in `environment_structural.json` and `environment_solver.json`:

```
cadquery-ocp==7.8.1.1
vtk==9.3.1
rtree==1.4.1
numpy==2.5.3
trimesh==4.12.2
jsonschema==4.26.0
usd-convert-cad==0.2.0
```

Install the native solver in a **separate** Python3.12 environment with `ovphysx==0.4.13` and `numpy==2.4.4`. Its bundled USD differs from the structural environment; importing `pxr` in the native solver process is unsupported. A Linux machine with the backend's supported runtime is required; the scored pilot used full L40 GPU hosts but the independent evaluator calls the CPU PhysX backend by default. These are recorded pins, not a claim that a fresh dependency resolution has been requalified on every platform.

Set paths for your machine (all commands below run on the Linux compute machine):

```sh
export EXP_ROOT=/absolute/path/to/astra-content-value-20260921
export STRUCT_PY=/absolute/path/to/repo/.venv/bin/python
export SOLVER_PY=/absolute/path/to/ovphysx-venv/bin/python
export GENERAL_DIR="$EXP_ROOT/evaluator/general"
export TMPDIR=/absolute/path/to/scratch
export UV_NO_CACHE=1
```

## Replay the frozen evaluation

The public bundle contains complete frozen inventories but does not redistribute the non-drawer reference NPZ files or most authored submissions. Rebuild and hash-align references as described below, or supply the separately retained originals. Only cases03/05 have a qualified byte-identical regeneration check. The following07 example therefore requires a separately staged complete reference directory and submission; it is not runnable from the public subset alone. The evaluator checks code hashes, inventory hashes, original source hashes, and every referenced NPZ digest before interpreting source fidelity. Relocate paths using command-line arguments; never edit frozen code or inventories to change a path.

```sh
"$STRUCT_PY" "$GENERAL_DIR/frozen/07_robot_arm/evaluate.py" \
  --case 07_robot_arm \
  --usd /absolute/path/to/submission/final.usd \
  --bindings /absolute/path/to/submission/bindings.json \
  --inventory "$GENERAL_DIR/references/07_robot_arm/source_inventory.json" \
  --source-root "$EXP_ROOT/assets/07_robot_arm" \
  --solver-python "$SOLVER_PY" \
  --device cpu \
  --output /absolute/path/to/new-evaluation
```

Cases03,05,06 and09 use the corresponding case names and asset roots. The source root passed to this evaluator is the whole case asset directory, even though the public task's authoring source closure may be a subdirectory such as `assets/09_printer/extracted`. `--structural-only` is diagnostic and never establishes functional acceptance. Normal evaluation launches fresh native processes for seeds11,23,47,83,131. It does not consume author-written trajectories or author pass claims.

## Rebuild references from originals

Reference regeneration is a separate reproducibility check, not permission to replace the frozen reference used to score the pilot. Tessellation ordering, archive serialization, absolute source-root metadata and package versions can affect byte hashes. Compare coverage, source IDs, bounds, topology counts and geometry digests; retain both the rebuilt and published reference. Any changed scoring reference requires a newly versioned experiment.

Copy only the mutable source-freezer helpers to a new directory with no `frozen/` child, then run:

```sh
export REBUILD_DIR=/absolute/path/to/new-reference-rebuild
mkdir -p "$REBUILD_DIR"
cp "$GENERAL_DIR/common.py" "$GENERAL_DIR/geometry.py" \
   "$GENERAL_DIR/freeze_dataset.py" "$GENERAL_DIR/finalize_roles.py" \
   "$GENERAL_DIR/case09_placement.py" "$GENERAL_DIR/case09_payload_contract.json" \
   "$REBUILD_DIR/"
nice -n 15 "$STRUCT_PY" "$REBUILD_DIR/freeze_dataset.py" \
  --dataset "$EXP_ROOT/protocol/dataset.json" --project "$EXP_ROOT" \
  --output "$REBUILD_DIR/references" \
  --case 03_hinge --case 05_vise --case 06_engine \
  --case 07_robot_arm --case 09_printer
"$STRUCT_PY" "$REBUILD_DIR/finalize_roles.py" 07_robot_arm
"$STRUCT_PY" "$REBUILD_DIR/finalize_roles.py" 09_printer
```

The STEP reader uses OCP7.8.1.1 and original occurrence placements; it tessellates placed solids with0.025mm chord tolerance and0.1rad angular tolerance. The FCStd reader opens saved ZIP/XML/BREP content and resolves saved linked documents without FreeCAD, macros or recomputation. FCStd references are cached part-local shape instances; they do not certify original mate transforms. glTF references use source node transforms and meter coordinates. Construction-only curves remain preserved in original files but are excluded from the surface inventory. All references record that scope.

Expected inventories:03 has8 mesh instances,05 has13 placed solids,06 has24 cached shape instances,07 has566 cached shape instances,09 has3092 placed solids. The large printer STEP took about30minutes to freeze in this environment and temporarily used approximately10GB RSS. Provision memory and scratch accordingly; do not run several such imports concurrently on a constrained host.

The independent fresh regeneration check recovered all13 case05 NPZ files byte-for-byte. Case03 requires one additional compatibility step: Trimesh creates random child-frame names for three bottle primitives, so their file order changes even though the complete eight-file SHA256 multiset is identical. Run `scripts/align_reference_rebuild_by_hash.py` against the unchanged complete frozen inventory and the fresh rebuild. It copies only exact regenerated bytes into hash-selected filenames and needs no original NPZ files. See `report/reference_regeneration.md` for the qualified commands and retained first, unaligned result. Do not rewrite the frozen importer, source IDs or expected hashes.

For printer payload placement, use exactly the five trusted bed-stack source IDs recorded in the published `case09_placement.json` as `--bed-source-id` arguments. The helper accepts repeated arguments and derives a clear flat source patch using25 support probes and collision/containment checks against overlapping original meshes. A shell-safe portable invocation is:

```sh
"$STRUCT_PY" - "$GENERAL_DIR" "$REBUILD_DIR" <<'PY'
import json, pathlib, subprocess, sys
original, rebuilt = map(pathlib.Path, sys.argv[1:])
bed_ids = json.loads((original/'case09_placement.json').read_text())['bed_source_ids']
cmd = [sys.executable, str(rebuilt/'case09_placement.py'),
       '--inventory', str(rebuilt/'references/09_printer/source_inventory.json'),
       '--output', str(rebuilt/'case09_placement.json')]
for source_id in bed_ids:
    cmd.extend(['--bed-source-id', source_id])
subprocess.run(cmd, check=True)
PY
```

## Qualification evidence and limits

The retained `tests/` directory contains native synthetic positives and negatives, including the six-joint arm, closed-loop engine, slider and three-axis printer with a free payload; the public subset includes selected qualification summaries rather than that entire directory. Case06 includes loaded/unloaded response. Case09 includes no-contact and locked-Z negatives. Case04's separate evidence establishes torque units and the important contact-unit correction: contact binding values in this pinned native API are impulses, so force is impulse divided by timestep. These fixtures qualify the observable/controller boundaries; they do not constitute solved benchmark-source submissions or certify hardware parameters.

Use the case snapshots, not mutable top-level evaluator code, when reproducing a scored result. The retained `local_archive_verification.json` records successful local verification of all five copied snapshots and their3703 geometry records. Case04, conveyor02 and cases08/10 have separate owners and qualification records. Infrastructure failure or unsupported observation is INCONCLUSIVE. A failed author claim becomes a demonstrated false pass only when the independent evaluator finds a concrete failing criterion.
