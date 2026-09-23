"""Native USD convex-mesh collision witness; synthetic box tray only."""
import json, subprocess
import drawer_evaluate as E
from drawer_test_fixtures import fixture,ROOT
from pxr import UsdGeom,UsdPhysics
s,b=fixture('synthetic_mesh_collision')
p='/World/Slider/Floor';s.RemovePrim(p);m=UsdGeom.Mesh.Define(s,p)
points=[(x,y,z) for x in [-.5,.5] for y in [.955,.965] for z in [-.18,.18]]
m.CreatePointsAttr(points)
faces=[0,1,3,2,4,6,7,5,0,4,5,1,2,3,7,6,0,2,6,4,1,5,7,3]
m.CreateFaceVertexCountsAttr([4]*6);m.CreateFaceVertexIndicesAttr(faces)
UsdPhysics.CollisionAPI.Apply(m.GetPrim());UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr('convexHull')
s.GetRootLayer().Save();out=ROOT/'run_mesh_collision';out.mkdir(exist_ok=True);E.trial_scene(s,b,11,out)
with (out/'solver.log').open('w') as log:subprocess.run(['/opt/astra-content-value-20260921/ovphysx-venv/bin/python',str(E.HERE/'drawer_solver.py'),str(out/'request.json')],stdout=log,stderr=subprocess.STDOUT,timeout=180)
r=json.loads((out/'trial_report.json').read_text());E.write_json(E.HERE/'mesh_collision_evidence.json',{'scope':'Synthetic convex-mesh USD collision runtime only; no scored source-cabinet trial','pass':r['status']=='PASS','trial':r});print(r['status'])
