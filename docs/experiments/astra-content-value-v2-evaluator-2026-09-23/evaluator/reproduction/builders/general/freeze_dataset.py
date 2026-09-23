"""Evaluator-owned source freeze. No physics authoring and no arm artifacts."""
import argparse
import json
from pathlib import Path
import subprocess
import traceback
import numpy as np
from common import dump,sha,transform
from geometry import freeze,mesh_record

def usd_reference(path, original, output):
    from pxr import Usd,UsdGeom
    stage=Usd.Stage.Open(str(path));scale=UsdGeom.GetStageMetersPerUnit(stage);cache=UsdGeom.XformCache();parts=[]
    for p in stage.Traverse():
        if not p.IsA(UsdGeom.Mesh):continue
        if UsdGeom.Imageable(p).ComputeVisibility()=='invisible':continue
        m=UsdGeom.Mesh(p);v=np.asarray(m.GetPointsAttr().Get(),float)
        v=transform(v,np.asarray(cache.GetLocalToWorldTransform(p),float).T)*scale
        idx=list(m.GetFaceVertexIndicesAttr().Get());faces=[];off=0
        for n in m.GetFaceVertexCountsAttr().Get():
            f=idx[off:off+n];faces.extend([[f[0],f[k],f[k+1]] for k in range(1,n-1)]);off+=n
        if not faces:continue
        f=np.asarray(faces);filename=f'geometry_{len(parts):04d}.npz';np.savez_compressed(output/filename,vertices=v,faces=f)
        parts.append({'source_id':str(original)+'#usd:'+str(p.GetPath()),'source_file':str(original),'frame':'source_assembly_world','geometry_file':filename,'geometry_sha256':sha(output/filename),'required':True,**mesh_record(v,f)})
    return {'parts':parts,'source_coverage_complete':False,'coverage_note':'Measured relative to independently frozen usd-convert-cad output. Native SolidWorks full assembly/dependency coverage is not independently verified; source fidelity therefore remains inconclusive.','converted_reference':str(path),'converted_reference_sha256':sha(path),'stage_meters_per_unit':scale,'importer_versions':{'usd-convert-cad':'0.2.0'}}

def main(a):
    dataset=json.loads(a.dataset.read_text());summary=[]
    for entry in dataset['assets']:
        case=entry['case_id']
        if case=='01_drawer' or (a.case and case not in a.case):continue
        output=a.output/case;output.mkdir(parents=True,exist_ok=True)
        assetroot=a.project/'assets'/case;primary=a.project/entry['primary_input'];originals=[]
        try:
            for original in entry['original_source_files']:
                p=a.project/entry['source_root']/original['path']
                if not p.is_file() or sha(p)!=original['sha256']:raise ValueError('Frozen original missing or changed: '+str(p))
                originals.append({'path':str(p.relative_to(assetroot)),'sha256':original['sha256']})
            unit=1 if primary.suffix.lower() in ('.glb','.gltf') else .001
            if primary.suffix.lower() in ('.sldasm','.sldprt'):
                reference=output/'reference.usdc'
                with (output/'converter.log').open('w') as log:
                    proc=subprocess.run([str(a.converter),'-i',str(primary),'-o',str(reference),'--up-axis','z','--instancing-style','none','--composition-style','none','--no-dedup','--convert-metadata'],stdout=log,stderr=subprocess.STDOUT,timeout=900)
                if proc.returncode!=0 or not reference.is_file():raise ValueError('Independent native conversion failed; see converter.log')
                inv=usd_reference(reference,primary.relative_to(assetroot),output)
            else:
                inv=freeze(assetroot,[primary.relative_to(assetroot)],unit,output,case)
            inv.update({'schema_version':1,'case_id':case,'source_root':str(assetroot),'source_files':originals,'dataset_sha256':sha(a.dataset),'source_closure_sha256':entry['source_closure_sha256'],'primary_input':str(primary.relative_to(assetroot))})
            role_map={}
            for part in inv['parts']:
                ident=part['source_id'];suffix=ident.split('#',1)[-1]
                if case=='03_hinge':
                    if suffix in ('node:FridgeDoor','node:FridgeDoorInner','node:FridgeGlass'):role_map[ident]='door'
                    if suffix=='node:FridgeCase':role_map[ident]='base'
                if case=='05_vise':
                    for token,role in [('Sliding Jaw','moving_jaw'),('Fixed Jaw','fixed_jaw'),('Base','base')]:
                        if token.lower() in suffix.lower():role_map[ident]=role
                if case=='06_engine':
                    for token,role in [('/piston001/','piston'),('/spoke_wheel_001/','crank'),('/machine_base_/','base'),('/beam_003/','beam'),('/spoke_wheel_rod_/','connecting_rod')]:
                        if token in suffix:role_map[ident]=role
                if case=='07_robot_arm' and '/AssemblyBase' in suffix:role_map[ident]='base'
            inv['source_body_role_requirements']=role_map
            dump(output/'source_inventory.json',inv)
            status={'case_id':case,'status':'frozen' if inv['source_coverage_complete'] else 'partial_reference_inconclusive','parts':len(inv['parts']),'inventory_sha256':sha(output/'source_inventory.json')}
        except Exception as exc:
            status={'case_id':case,'status':'source_ingestion_inconclusive','error':str(exc),'traceback':traceback.format_exc()}
        dump(output/'freeze_status.json',status);summary.append(status);print(json.dumps(status),flush=True)
    dump(a.output/'freeze_summary.json',summary)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--case',action='append');p.add_argument('--converter',type=Path,default=Path('/opt/astra-content-value-20260921/repo/.venv/bin/usd-convert-cad'));main(p.parse_args())
