# OpenVDB native wheel provenance

The `openvdb` native wheel is built from the Academy Software Foundation's
OpenVDB `v13.0.0` release commit. `source-lock.json` pins the commit, immutable
source archive digest, source-tree inventory digest, patch inventory, Python ABI
and build-tool versions. It also pins the complete compiled-input source chain:
the OpenEXR Project Half implementation embedded in OpenVDB, oneTBB 2022.2.0,
c-blosc 1.21.6 and its enabled vendored codecs (including the
libdivsufsort-lite copy inside Zstandard), zlib 1.3.1, nanobind 2.13.0, and
nanobind's robin-map 1.4.0 submodule. The lock records exact paths and SHA-256
values for both Half source files and both libdivsufsort-lite source files. The
build refuses an archive, source tree, license, patch, or overlay whose digest
differs from that lock.

The build copies the project overlay first and then applies every numbered patch
in lexical order. The patches keep the official `NB_MODULE(openvdb, ...)`
extension, add the content-agent tools to that module, correct upstream's stale
wheel metadata, and make wheel RPATHs relocatable. No subprocess adapter or
second Python extension is introduced.

The native dependencies are built in a private prefix. c-blosc and zlib are
linked statically; oneTBB is the only non-platform shared dependency. OpenVDB's
CMake cache and the wheel's ELF closure are checked against that layout, so a
system package with a matching SONAME cannot silently satisfy the build.

oneTBB's `third-party-programs.txt` item 4 describes a workaround for GCC bug
62258 that accesses the old libstdc++ internal exception ABI. That code is
guarded for libstdc++ versions from GCC 4.7 through GCC 5. The locked GCC 11
toolchain evaluates the guard to false, so only oneTBB's inert fallback stubs
are compiled. The build verifies that neither `exception.cpp.o` nor the final
`libtbb.so` contains the workaround's `__cxa_get_globals` symbol and records the
excluded source feature in native dependency provenance.

The release artifact is repaired with `auditwheel`, installed without dependencies
into a clean CPython 3.12 environment, imported with the build prefix removed
from the library search path, and inspected for OpenVDB ABI 13 without NumPy
installed or loaded. A deterministic native dependency provenance file binds source,
license, and built-artifact SHA-256 values inside the wheel. The sidecar build
manifest records that evidence plus the wheel, native members, toolchain, and
runtime library closure used for the invocation.
