"""Opt-in synthetic backend smoke; no model, benchmark source, or solver."""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
 if not a.execute:raise SystemExit('Explicit --execute required; synthetic optimizer/repair execution only.')
 out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
 from pxr import Usd,UsdGeom
 source=out/'synthetic.usda';s=Usd.Stage.CreateNew(str(source));root=UsdGeom.Xform.Define(s,'/Fixture');s.SetDefaultPrim(root.GetPrim());UsdGeom.SetStageMetersPerUnit(s,1);UsdGeom.SetStageUpAxis(s,'Z');UsdGeom.Cube.Define(s,'/Fixture/Box');s.GetRootLayer().Save()
 original=hashlib.sha256(source.read_bytes()).hexdigest();checks={};errors=[]
 try:
  from world_understanding.functions.graphics.scene_optimizer_local import optimize_usd_local
  optimized=out/'optimized.usda';r=optimize_usd_local(source,optimized,{'timeout':120,'scene_optimizer_settings':{'enable_deinstance':True,'enable_split_meshes':False,'enable_deduplicate':False}},approved_dependency_roots=[out]);(out/'optimizer.json').write_text(json.dumps(r,indent=2,default=str)+'\n');stage=Usd.Stage.Open(str(optimized));checks['optimizer_actual_output']=bool(stage and stage.GetPrimAtPath('/Fixture/Box'))
 except Exception as e:errors.append({'step':'optimizer','error':repr(e)})
 try:
  from geometry_repair.workers.geogram_local_repair import _verified_vorpalite
  exe,sha,reason=_verified_vorpalite(expected_version='1.10.0');assert exe,reason
  obj=out/'tetra.obj';obj.write_text('v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n');target=out/'tetra_repaired.obj'
  cmd=[str(exe),str(obj),str(target),'profile=repair','pre=true','pre:repair=true','pre:intersect=false','pre:epsilon=0%','pre:max_hole_area=0%','pre:max_hole_edges=0','pre:min_comp_area=0%','pre:remove_internal_shells=false','remesh=false','post=false','sys:max_threads=1','log:quiet=true']
  r=subprocess.run(cmd,capture_output=True,text=True,timeout=120);(out/'geogram.stdout').write_text(r.stdout);(out/'geogram.stderr').write_text(r.stderr)
  import trimesh,numpy as np
  repaired=trimesh.load(target,process=False);checks['geogram_actual_output']=bool(r.returncode==0 and len(repaired.faces)==4 and np.isfinite(repaired.vertices).all());checks['geogram_verified_sha256']=sha
 except Exception as e:errors.append({'step':'geogram','error':repr(e)})
 try:
  from world_understanding.functions.graphics.validate_usd import validate_usd
  r=validate_usd(source,fix=False);(out/'usd_validator.json').write_text(json.dumps(r,indent=2,default=str)+'\n');checks['usd_validator_executed']=r.get('status')=='success'
 except Exception as e:errors.append({'step':'usd_validator','error':repr(e)})
 checks['source_unchanged']=hashlib.sha256(source.read_bytes()).hexdigest()==original
 passed=not errors and all(checks.get(k) is True for k in ['optimizer_actual_output','geogram_actual_output','usd_validator_executed','source_unchanged'])
 receipt={'schema_version':'matched-environment-synthetic-backend-smoke.v2','passed':passed,'checks':checks,'errors':errors,'models_launched':False,'solver_or_renderer_launched':False,'full_native_stage_chain_qualified':False,'scope':'Synthetic optimizer/repair/validator availability only; does not establish task acceptance or full native workflow completion.'};(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));return 0 if passed else 2
if __name__=='__main__':raise SystemExit(main())
