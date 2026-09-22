# Frozen drawer submission and acceptance contract

Submit a self-contained `final.usda` (or `final.usd`/`final.usdc`) with its dependencies and `bindings.json`. Keep Y up, metres, the original closed component geometry in source world coordinates, and preserve all five original component surfaces. Local body origins may change if composed world geometry stays the same. Mesh reindexing/splitting is allowed; simplified primitive replacements for the source visuals are rejected.

`bindings.json`:

```json
{
  "asset_id": "01_drawer",
  "final_usd": "final.usda",
  "drawer_body": "/World/Drawer01",
  "cabinet_body": "/World/Cabinet",
  "drawer_joint": "/World/DrawerJoint",
  "source_components": {
    "drawer_cabinet": ["/World/Cabinet/Visual"],
    "drawer_cabinet_drawer_01": ["/World/Drawer01/Visual"],
    "drawer_cabinet_drawer_02": ["/World/Drawer02/Visual"],
    "drawer_cabinet_drawer_03": ["/World/Drawer03/Visual"],
    "drawer_cabinet_drawer_04": ["/World/Drawer04/Visual"]
  }
}
```

Use absolute USD prim paths. The strings above are examples, not required naming. The cabinet is anchored as static collision geometry (no enabled rigid body) and the upper source drawer is a nonkinematic `PhysicsRigidBodyAPI` body with explicit positive mass (0.5–10kg), diagonal inertia, and center of mass. The other drawers remain static/anchored. A standalone `PhysicsPrismaticJoint` must connect the static/world frame as body0 and target drawer as body1, be enabled, have a world axis parallel to positive Z, lower limit within ±0.002m, and upper limit 0.22–0.45m. The joint may use any local axis with matching local frame rotations. Do not use an articulation root for this one-joint contract; direct rigid-body force observation is the common backend.

Retain original source meshes as visible render geometry. Provide enabled collision geometry for the cabinet and moving drawer. Approximation is allowed, including component cuboids/convex pieces, but the drawer interior and cabinet opening must stay clear. Do not use a single filled convex hull/box over either opening, hidden copies as a substitute for source visuals, or collision filtering that removes physical drawer/payload contact. No geometry/physics time samples, preauthored trajectories, fixed payload attachment, scripts, or custom controller callbacks. The evaluator ignores any submitted success files and trajectories.

The evaluator imports the actual submission into ovphysx0.4.13, adds its own free 0.5kg box, sets gravity, removes authored joint drives, and applies a bounded external force to the drawer. Force-PD gains, 40N cap, ramp timings, perturbations, five seeds, and every tolerance are frozen in `drawer_acceptance.json`. No body pose or velocity is written during stepping. Drawer opening must reach 0.20m, close within 0.02m, keep the payload in the drawer, avoid >5mm reported penetration, and remain finite/stable in every seed. Source fidelity compares all five components' triangles, vertex and centroid distances, area, bounds, and welded shape closure against the untouched publisher mesh. Interior sphere queries independently detect filled collision approximations.

Outputs: input/source/evaluator hashes, `structural_report.json`, immutable per-seed `scene.usda`, solver logs, per-step pose/velocity/applied-force/contact JSONL, `trial_report.json`, replay USD time samples generated **after** the solver run, and aggregate `report.json`. Replay animation is an output visualization, not solver input or a scoring oracle. The same evaluator and protocol apply to both arms.

Run with the regular repository USD Python environment; the command starts the separate isolated ovphysx Python itself:

```
/opt/astra-content-value-20260921/repo/.venv/bin/python drawer_evaluate.py --bindings /absolute/submission/bindings.json --output /absolute/evidence/run
```

The default solver path is `/opt/astra-content-value-20260921/ovphysx-venv/bin/python`. All simulations run on the assigned Horde host. The evaluator's synthetic-fixture tests are not source-cabinet benchmark runs and cannot establish asset success.
