# Geogram Worker Third-Party Notices

The Geometry Repair Geogram worker builds `vorpalite` from Geogram v1.10.0,
commit `c8529bb00838186938ab31d96008a59b6a892dee`, under BSD-3-Clause. This is
the exact release-tag commit used by both CAD Agent container builds and
the runtime version gate.

The pinned Geogram tree records these recursively fetched submodules:

| Component | Commit | License |
| --- | --- | --- |
| OpenNL | `ae782db2db40cc40fcd7659629fc652e5eab1f9e` | BSD-3-Clause |
| AMGCL | `93827a00fc926d951c75f08fdb1d491912ff7065` | MIT |
| pybind11 (AMGCL submodule) | `ffa346860b306c9bbfb341aed9c14c067751feb8` | BSD-3-Clause |
| libMeshb | `88095d5ed04bfaf6bee27b777e7190bebdc72230` | MIT |
| rply | `4296cc91b5c8c26d4e7d7aac0cee2b194ffc5800` | MIT |
| ImGuiColorTextEdit (graphics disabled) | `062408b1887534b9585e7cf3410ae57e3ebe7497` | MIT/BSD-3-Clause |
| Dear ImGui (graphics disabled) | `01a4cff8f9190b7f3731e4d1cef818ca675e9a42` | MIT |
| ImGui Lua bindings (graphics/Lua disabled) | `33cf2501a01ed51e8ad867eda227cb3917011f1c` | BSD-3-Clause |
| imoguizmo (graphics disabled) | `a05c47fb2c99e2d40858f7e0a4f6d48490747af9` | MIT |
| GLFW (graphics disabled) | `e7ea71be039836da3a98cea55ae5569cb5eb885c` | Zlib |

The selected non-graphics build also compiles permissively licensed source
vendored in the Geogram tree: PoissonRecon (BSD-3-Clause), zlib (Zlib),
xatlas (MIT), and stb/stb_image (public-domain or MIT at the author's option).
Lua, TetGen, Triangle, HLBFGS, graphics, and legacy numerics are disabled.
The reviewed build patch also makes `GEOGRAM_WITH_TBB=OFF` disable Geogram's
unconditional Linux parallel-STL link. Geogram therefore uses its sequential
STL path and adds no oneTBB build or runtime dependency.

The image retains Geogram's license, this notice, the exact recursive
submodule status, the tag/commit provenance file, and available upstream
license files under `/opt/geogram/share/licenses`.
