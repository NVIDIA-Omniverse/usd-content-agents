# Agentic Workflow Examples

Examples are grouped by the workflow they launch. This catalog contains the
public examples; additional private examples are excluded from public staging.

| Workflow | Example | Asset |
|---|---|---|
| Geometry | [`geometry/quickstart`](geometry/quickstart/) | checked-in smoke bracket |
| Mesh segmentation | [`mesh-segmentation/basic`](mesh-segmentation/basic/) | checked-in fused cart |
| Materials | [`materials/basic-assignment`](materials/basic-assignment/) | existing ladder and references |
| Physics | [`physics/basic`](physics/basic/) | existing Lightbulb01 package |
| Physics tune/refine | [`physics/tire-bounce`](physics/tire-bounce/) | existing Tire_B01 package and behavior-guided PNG evidence |
| Physics refine | [`physics/container-slide`](physics/container-slide/) | existing Container_Gray_C04 package |
| SimReady | [`simready/validate-and-conform`](simready/validate-and-conform/) | existing SimReady cube |
| Scene | [`scene/material-pass`](scene/material-pass/) | pinned public NVIDIA SimReady workcell assets |
| CAD-to-SimReady | [`cad-to-simready/toycar`](cad-to-simready/toycar/) | pinned public ToyCar download |
| Texture | [`texture/basic`](texture/basic/) | existing UV-ready ladder |
| Articulation | [`articulation/drawer`](articulation/drawer/) | provider-optional checked-in mini drawer cabinet |
| Validation | [`validation/basic`](validation/basic/) | checked-in smoke bracket |

The examples reuse assets in their owning packages instead of copying them.
Only small USDA fixtures are stored here. ToyCar and the Scene material pass
provide pinned public download helpers for assets that do not ship in this
repository.

Run commands from the repository root after completing the setup in
[`../README.md`](../README.md).

Native Windows execution is unsupported in the 0.6 release. On a Windows host,
run the Bash launchers inside WSL2. The checked-in PowerShell `run.ps1` and
`fetch.ps1` helpers are retained for development and future qualification; they
are not supported 0.6 setup or execution paths.
