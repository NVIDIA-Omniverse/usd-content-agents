# OpenVDB source patch order

The wheel build must prepare the pinned OpenVDB `v13.0.0` source tree in this
order:

1. Copy `native/overlay/` over the OpenVDB source root. This installs
   `openvdb/openvdb/python/pyTools.cc`.
2. Apply numbered patches from this directory with `git apply`.
3. Configure OpenVDB with `OPENVDB_BUILD_PYTHON_MODULE=ON`.

`0001-export-python-tools.patch` deliberately adds `pyTools.cc` to the upstream
`openvdb_python` nanobind target and invokes `exportTools()` from its existing
`NB_MODULE(openvdb, ...)` entry point. It must not be compiled as a second
extension module.

`0003-relocatable-wheel-and-source-lock.patch` links wheel targets with their
final `$ORIGIN` RPATH. This prevents absolute build-root length from changing
ELF padding during installation and keeps repaired wheel bytes reproducible.

`0004-bounded-volume-to-mesh.patch` adds caller-supplied point and primitive
ceilings to `VolumeToMesh`. It rejects exact point counts and conservative
primitive upper bounds before the corresponding output arrays are allocated.
