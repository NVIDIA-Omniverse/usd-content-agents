# Third-party notices for the OpenVDB native wheel

The native wheel is derived from these projects:

| Component | Use | License |
| --- | --- | --- |
| OpenVDB 13.0.0 | Sparse volume library and official Python module | Apache-2.0 |
| OpenEXR Project Half implementation, OpenVDB 13.0.0 embedded copy | `Half.cc` is compiled into OpenVDB and `Half.h` is shipped in the wheel | BSD-3-Clause |
| oneTBB 2022.2.0 | Source-built shared parallel runtime dependency | Apache-2.0 |
| c-blosc 1.21.6 | Source-built static VDB compression dependency | BSD-3-Clause |
| zlib 1.3.1 | Source-built static VDB compression dependency | Zlib |
| LZ4 1.9.4 | Compiled into c-blosc from its locked source archive | BSD-2-Clause |
| Zstandard 1.5.6 | Compiled into c-blosc from its locked source archive | BSD-3-Clause (selected from the upstream BSD-3-Clause or GPL-2.0-only offer) |
| libdivsufsort-lite, Zstandard 1.5.6 vendored copy | `divsufsort.c` is compiled into c-blosc's static archive | MIT |
| Bitshuffle adapted source | Compiled into c-blosc from its locked source archive | MIT |
| FastLZ-derived source | Compiled into c-blosc from its locked source archive | MIT |
| zlib-ng-derived fastcopy source | Compiled into c-blosc from its locked source archive | Zlib |
| nanobind 2.13.0 | Compiled into the Python extension from locked source | BSD-3-Clause |
| robin-map 1.4.0 | Header-only nanobind input from its locked source | MIT |

The verified OpenVDB source archive contributes its full `LICENSE` file to the
wheel metadata. The OpenEXR Half and libdivsufsort-lite entries are tracked
separately because their source-file licenses differ from their containing
projects' headline licenses. Full redistribution terms for each compiled input
are stored under `native/licenses/` and copied into the wheel license directory. c-blosc
1.21.6 does not contain Snappy source, so this build explicitly disables Snappy
and neither compiles nor redistributes it. `auditwheel` decides which
non-system shared libraries are bundled; the source-locked native license
policy and generated admission report are the
authoritative component, archive-member, and ELF dependency inventories for
each artifact. The build fails on unknown inventory entries and on introduced,
bundled, or non-platform runtime components under LGPL, GPL, or AGPL terms.

oneTBB's retained upstream notice includes item 4 for an old libstdc++ exception
workaround with a GCC internal ABI caveat. The workaround is guarded to GCC
libstdc++ versions 4.7 through 5 and is not compiled by this wheel's locked GCC
11 toolchain. The build requires its identifying `__cxa_get_globals` symbol to
be absent from both the oneTBB exception object and the bundled `libtbb.so`.

Non-bundled glibc and GCC runtime libraries are recorded as the external Linux
and compiler platform ABI. Application-owned array providers, including NumPy
where an application already selects it, are neither installed by this native
wheel build nor part of its introduced dependency closure. That explicit
boundary does not claim that a base image or the application's pre-existing
Python dependency closure is free of LGPL, GPL, or AGPL components.
