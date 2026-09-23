"""Two small synthetic native CPU traces in the same read-only /tools namespace.
Gravity-on must fall; gravity-zero is the negative control for that predicate.
No benchmark source, model, GPU, or workflow acceptance is involved.
"""
import hashlib,json,time
from pathlib import Path
from pxr import Usd,UsdGeom,UsdPhysics,Gf
from usd_core.physics_runtime import simulate_scene
def main():
 out=Path('/work');started=time.time();results=[]
 for name,gravity in [('gravity_on',9.81),('gravity_zero',0.0)]:
  source=out/(name+'.usda');stage=Usd.Stage.CreateNew(str(source));root=UsdGeom.Xform.Define(stage,'/Fixture');stage.SetDefaultPrim(root.GetPrim());UsdGeom.SetStageMetersPerUnit(stage,1.0);UsdGeom.SetStageUpAxis(stage,'Z')
  scene=UsdPhysics.Scene.Define(stage,'/Fixture/PhysicsScene');scene.CreateGravityDirectionAttr(Gf.Vec3f(0,0,-1));scene.CreateGravityMagnitudeAttr(gravity)
  cube=UsdGeom.Cube.Define(stage,'/Fixture/Body');cube.CreateSizeAttr(0.2);cube.AddTranslateOp().Set(Gf.Vec3d(0,0,1));UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim());UsdPhysics.CollisionAPI.Apply(cube.GetPrim());UsdPhysics.MassAPI.Apply(cube.GetPrim()).CreateMassAttr(1.0);stage.GetRootLayer().Save();before=hashlib.sha256(source.read_bytes()).hexdigest()
  report=simulate_scene(source,str(out/name),body_path='/Fixture/Body',rest_position=[0,0,1],world_up=[0,0,1],duration_s=0.2,dt=1/240,sample_fps=30)
  rows=[json.loads(line) for line in Path(report['trajectory_jsonl']).read_text().splitlines() if line.strip()];z=rows[-1]['pose'][2] if rows else None
  falls=z is not None and z<0.9
  correct=falls if gravity else z is not None and abs(z-1)<1e-5 and not falls
  results.append({'case':name,'gravity':gravity,'final_z':z,'falls_predicate':falls,'expected_falls':bool(gravity),'passed':correct and report['n_bodies']==1 and report['simulation_facts']['trajectory_finite'],'steps':report['simulation_facts']['reported_step_count'],'samples':len(rows),'source_sha256':before,'source_unchanged':hashlib.sha256(source.read_bytes()).hexdigest()==before,'runtime_report_sha256':hashlib.sha256(Path(report['report_path']).read_bytes()).hexdigest(),'trace_sha256':hashlib.sha256(Path(report['trajectory_jsonl']).read_bytes()).hexdigest()})
 receipt={'schema_version':'readonly-native-physics-functional-smoke.v2','passed':all(r['passed'] and r['source_unchanged'] for r in results),'results':results,'elapsed_seconds':time.time()-started,'native_solver':'OvPhysX0.4.13 CPU','model_calls':False,'gpu_jobs':False,'benchmark_sources_used':False,'end_to_end_workflow_qualified':False,'scope':'Only synthetic gravity dynamics and negative control through installed native simulate_scene in read-only namespace; no task or workflow acceptance.'}
 (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));return 0 if receipt['passed'] else 2
if __name__=='__main__':raise SystemExit(main())
