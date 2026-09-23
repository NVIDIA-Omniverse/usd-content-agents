"""Original3092-mesh USD readback benchmark; no articulated authoring/solution."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
from pxr import Sdf,Usd,UsdGeom,Vt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'cases/09_printer'))
from structural import mesh_world
from v2_geometry import surface_compare
from v2_process import inspect_with_deadline

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def measure(scene,reference,inventory,output):
    start=time.monotonic();stage=Usd.Stage.Open(str(scene));opened=time.monotonic();passed=True;count=0
    with (output/'parts.jsonl').open('w') as log:
        for i,part in enumerate(inventory['parts']):
            path=reference/part['geometry_file'];assert sha(path)==part['geometry_sha256']
            with np.load(path) as a:
                v,f=mesh_world(stage.GetPrimAtPath('/Source/Part'+str(i)))
                tol=max(5e-5,.001*float(np.linalg.norm(np.ptp(a['vertices'],axis=0))))
                okay,detail=surface_compare(a['vertices'],a['faces'],v,f,tol)
                passed &= okay;count+=1
                log.write(json.dumps({'source_id':part['source_id'],'passed':okay,'measurement':detail})+'\n')
    return {'passed':bool(passed),'parts_checked':count,'source_stage_open_s':opened-start,'stage_open_and_all_mesh_readback_compare_s':time.monotonic()-start}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--existing-source-only-scene',type=Path);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    ref=ROOT/'cases/09_printer/reference';invpath=ref/'source_inventory.json';inventory=json.loads(invpath.read_text());scene=a.output/'original_source_visual_only.usdc'
    if a.existing_source_only_scene:
        result=inspect_with_deadline(measure,a.existing_source_only_scene,ref,inventory,a.output)
        result.update(scope='Fresh Python process loads source-only USD from disk and checks every original part using actual mesh_world/comparator; no simulator or task acceptance.',source_inventory_sha256=sha(invpath),source_stage_sha256=sha(a.existing_source_only_scene),comparison_code_sha256=sha(ROOT/'cases/09_printer/v2_geometry.py'),structural_code_sha256=sha(ROOT/'cases/09_printer/structural.py'),script_sha256=sha(Path(__file__)),model_calls=0)
        (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));return
    start=time.monotonic();stage=Usd.Stage.CreateNew(str(scene));UsdGeom.SetStageMetersPerUnit(stage,1);UsdGeom.SetStageUpAxis(stage,'Z');UsdGeom.Xform.Define(stage,'/Source')
    with Sdf.ChangeBlock():
        # Use low-level prim specs in one change block; no runtime bodies,
        # colliders, joints, role bindings or benchmark solution are authored.
        for i,part in enumerate(inventory['parts']):
            path=ref/part['geometry_file'];assert sha(path)==part['geometry_sha256']
            with np.load(path) as data:
                prim=Sdf.CreatePrimInLayer(stage.GetRootLayer(),'/Source/Part'+str(i));prim.specifier=Sdf.SpecifierDef;prim.typeName='Mesh'
                for name,typ,value in [('points',Sdf.ValueTypeNames.Point3fArray,Vt.Vec3fArray.FromNumpy(data['vertices'].astype(np.float32))),('faceVertexCounts',Sdf.ValueTypeNames.IntArray,Vt.IntArray.FromNumpy(np.full(len(data['faces']),3,np.int32))),('faceVertexIndices',Sdf.ValueTypeNames.IntArray,Vt.IntArray.FromNumpy(data['faces'].reshape(-1).astype(np.int32)))]:Sdf.AttributeSpec(prim,name,typ).default=value
    stage.GetRootLayer().Save();stage=None;build_s=time.monotonic()-start
    result=inspect_with_deadline(measure,scene,ref,inventory,a.output)
    result.update(scope='All original source visual meshes serialized to USD and read by actual evaluator mesh_world and fidelity comparator. No joints, bodies, colliders, solver, authored output or task acceptance.',source_inventory_sha256=sha(invpath),source_stage_sha256=sha(scene),source_stage_bytes=scene.stat().st_size,source_build_s=build_s,comparison_code_sha256=sha(ROOT/'cases/09_printer/v2_geometry.py'),structural_code_sha256=sha(ROOT/'cases/09_printer/structural.py'),model_calls=0)
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
