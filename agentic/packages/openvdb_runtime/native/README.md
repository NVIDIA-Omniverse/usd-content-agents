# OpenVDB 13 native wheel

`build_openvdb_wheel.sh` builds the official in-process `openvdb` nanobind
module for CPython 3.12. It verifies the pinned source archive and reconstructed
Git-tree inventory, copies the NVIDIA tool overlay, applies the numbered patches,
builds exact-source zlib, c-blosc, and oneTBB dependencies, repairs the wheel,
and imports it in a clean environment with `LD_LIBRARY_PATH` and `PYTHONPATH`
unset. Ambient compiler, linker, CMake, package-discovery, and library-loader
environment overrides are rejected before preparation. The clean-environment
smoke installs only the built wheel, verifies ABI 13 and the tools API, and uses
a real grid without installing or importing NumPy. Full numeric and mesh
coverage runs in application test environments that independently own their
array provider. `scalar_mean` is limited to non-level-set FloatGrids; level sets
must use `level_set_mean` so their grid-class invariant remains valid.

The source lock defines `manylinux_2_36` as the requested auditwheel repair
target, not as a ban on compatible lower-version tags. Auditwheel may add tags
such as `manylinux_2_34` when the inspected ELF closure supports them; the
promoted record binds the complete resulting filename. After repair, the build
canonicalizes ZIP member order, metadata, and compression with the locked
CPython zlib runtime so identical wheel contents produce identical archive
bytes across qualified builders. Qualified artifacts are built on the locked
Ubuntu 22.04 x86_64 toolchain and consumed by the Debian 12 production image:

```bash
agentic/packages/openvdb_runtime/native/build_openvdb_wheel.sh
```

The wheel and its build, prepromotion, final-admission, provenance, release-lock,
and `ldd` evidence are written to `dist-native/` unless
`OPENVDB_RUNTIME_WHEEL_DIR` selects another directory. Use `--prepare-only` to
verify and patch sources without compiling.

A release build succeeds only when the repaired wheel matches the `promoted`
record for its architecture in `release-lock.json`. The record binds exact
intermediates and every native member independently of build-generated
provenance. `generate_release_candidate.py` can emit deterministic candidate
evidence for review; it cannot write the lock or emit `promoted` status, and its
output is deliberately rejected by admission until reviewed and checked in.
`build_openvdb_wheel.sh --candidate` runs all structural, license, and ELF
prepromotion checks and emits that non-deployable review bundle without running
final admission.

The oneTBB source patch disables ITT notification loading and optional allocator,
affinity, RML, TCM, and OpenMP helper discovery. Build-time ELF assertions reject
the corresponding library names and dynamic-loader symbols in `libtbb`.

`OPENVDB_RUNTIME_AUDITWHEEL_PLAT` exists only for validating a local toolchain.
A wheel built with a platform override is not a release artifact. Production
builds use the source-locked default, and the requested target plus actual wheel
identity are recorded in the native build manifest.
