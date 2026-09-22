# Frozen reference regeneration check:03 and05

Both reference sets can be reconstructed byte-for-byte in the recorded pinned Linux runtime without distributing their NPZ files, provided the full frozen inventories are published. Case03 needs the explicit hash-alignment step below. Directly rerunning the freezer is insufficient for that case.

| Case | Raw fresh rebuild | Exact arrays | Compatible with unchanged frozen inventory |
|---|---|---|---|
|03 refrigerator glTF,8 parts|Five filenames matched; the three bottle files were permuted|All8 match exactly after identifying files by expected SHA256|Yes after copying unchanged regenerated files into the hash-selected frozen filenames|
|05 vise STEP,13 solids|All13 NPZ files matched at their original filenames|All13 exactly equal|Yes directly|

Case03 ran from23:21:12 to23:21:20UTC on2026-09-21; source import took6.83s. Case05 ran from23:21:48 to23:24:07UTC; import took138.73s. Both used nice15, idle I/O, one CPU affinity, CUDA hidden and single-thread numerical-library settings. These are loaded-host observations, not performance estimates. The source closure, frozen evaluator code and original inventories remained unchanged.

The complete original inventory hashes are:

- 03: `c4fcc67778174cc387d847063096de1a8ca071eaea05fe12e627e31f5b1f55b3`
- 05: `76c100c81816bce57e16b65f3c5dd37cc71f79f94bb152a9be17f34421bb662f`

The pinned dependencies were NumPy2.5.3, Trimesh4.12.2 and cadquery-ocp7.8.1.1. `launch.json` records Python, platform, zlib, CPU affinity, commands, source hashes and every frozen-code hash. No fresh package installation or source redownload was tested. Results do not establish cross-version or cross-platform byte stability.

## Why03 needs alignment

Trimesh's glTF loader creates child frames for nodes with multiple mesh primitives. At `trimesh/exchange/gltf/__init__.py:1920`, it appends `util.unique_id(length=6)` to the frame name. `trimesh/util.py:2197` derives that identifier from random128-bit values. The frozen freezer sorts the resulting frame names. Three `ChampagneBottles` primitive names therefore changed, and their exact geometry files exchanged positions. The five refrigerator component files did not change.

This was neither a geometry difference nor a ZIP serialization difference: the rebuilt multiset of eight complete NPZ SHA256 values was identical. All vertex/face arrays, shapes and dtypes were separately compared after matching. The saved first raw rebuild remains marked incompatible; the separate `aligned/` result records the exact filename mapping and unchanged inventory. A second `public_compatible/` check used an inventory-only directory containing no original NPZ files, confirming that public regeneration does not require retained reference geometry.

Observed installed implementation hashes:

- Trimesh glTF loader: `2cb526d86179ab36d6b475ac6b9909935811661af4b920ff0f83b58c6c84b558`
- Trimesh utility module: `1a1a7ea54c998d3259df469118e2d6e7dc1093e0e2fbffb279c4e8d2443ec668`

## Portable regeneration

First fetch and hash-verify the exact original sources using the published source manifests. Install the recorded structural dependency pins described in `evaluator/general/REBUILD.md`. Publish `evaluator/general/references/<case>/source_inventory.json` and each `frozen/<case>/frozen_manifest.json` unchanged. The author-facing `protocol/tasks/*_source_inventory.json` omits geometry filenames/digests and cannot substitute for the complete frozen inventory.

Run on the Linux compute machine, choosing a fresh output directory:

```sh
export REPRO_ROOT=/absolute/path/to/astra-content-value-20260921
export STRUCT_PY=/absolute/path/to/pinned-structural-environment/bin/python
export REBUILD_DIR=/absolute/path/to/new-reference-rebuild
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=''
REPRO_CPU=$("$STRUCT_PY" -c 'import os; print(max(os.sched_getaffinity(0)))')

nice -n15 ionice -c3 taskset -c "$REPRO_CPU" "$STRUCT_PY" \
  "$REPRO_ROOT/evaluator/general/frozen/03_hinge/geometry.py" \
  --asset-root "$REPRO_ROOT/assets/03_hinge" \
  --source source/CommercialRefrigerator.glb --unit 1 \
  --output "$REBUILD_DIR/03_raw" --case 03_hinge

nice -n15 ionice -c3 taskset -c "$REPRO_CPU" "$STRUCT_PY" \
  "$REPRO_ROOT/scripts/align_reference_rebuild_by_hash.py" \
  --inventory "$REPRO_ROOT/evaluator/general/references/03_hinge/source_inventory.json" \
  --rebuilt "$REBUILD_DIR/03_raw" --output "$REBUILD_DIR/03_compatible"

nice -n15 ionice -c3 taskset -c "$REPRO_CPU" "$STRUCT_PY" \
  "$REPRO_ROOT/evaluator/general/frozen/05_vise/geometry.py" \
  --asset-root "$REPRO_ROOT/assets/05_vise" \
  --source 'source/Machine Vice  Assembly.STEP' --unit 0.001 \
  --output "$REBUILD_DIR/05_raw" --case 05_vise

nice -n15 ionice -c3 taskset -c "$REPRO_CPU" "$STRUCT_PY" \
  "$REPRO_ROOT/scripts/align_reference_rebuild_by_hash.py" \
  --inventory "$REPRO_ROOT/evaluator/general/references/05_vise/source_inventory.json" \
  --rebuilt "$REBUILD_DIR/05_raw" --output "$REBUILD_DIR/05_compatible"
```

The helper fails closed unless the complete expected and regenerated NPZ hash multisets match. It only copies files and the original inventory; it never serializes arrays, changes a digest or edits a frozen source file. Its default mode needs no original NPZ files. `--compare-original-arrays` adds the optional deep comparison when original NPZ files are retained. The independent diagnostic script `check_reference_rebuild.py` performed that deeper comparison for this qualification and saved the raw mismatch.

Pass the compatible inventory path to the unchanged evaluator, alongside the original source root and separately supplied submission. This check establishes reference byte compatibility, not a rerun of authored physics. No author outputs were read.

Evidence: `repro-check/reference-regeneration-v1/{03_hinge,05_vise}/result.json`, corresponding `launch.json`, raw rebuild logs, and03's separate alignment receipts. Do not generalize this qualification to the remaining larger CAD cases without checking their regenerated hashes.
