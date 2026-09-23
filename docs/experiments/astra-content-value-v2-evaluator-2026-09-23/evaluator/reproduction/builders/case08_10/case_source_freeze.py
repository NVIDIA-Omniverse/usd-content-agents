"""Independent source selection. Never authors physics or benchmark solutions."""
import io,json,shutil,zipfile
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from common import dump,sha
from geometry import occ_mesh,mesh_record

ROOT=Path('/opt/astra-content-value-20260921')
HERE=Path(__file__).resolve().parent

def hand():
    from OCP.BRep import BRep_Builder
    from OCP.BRepTools import BRepTools
    from OCP.TopoDS import TopoDS_Shape
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_SHELL,TopAbs_FACE
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    original=json.loads((ROOT/'evaluator/general/references/10_complex/source_inventory.json').read_text())
    asset=Path(original['source_root']);rel=Path(original['parts'][0]['source_file']);assembly=asset/rel
    output=HERE/'references/10_complex';output.mkdir(parents=True,exist_ok=True)
    def shapes(path):
        with zipfile.ZipFile(path) as z:
            doc=ET.fromstring(z.read('Document.xml'))
            for obj in doc.find('ObjectData'):
                part=obj.find("Properties/Property[@name='Shape']/Part")
                if part is None:continue
                shape=TopoDS_Shape();BRepTools.Read_s(shape,io.BytesIO(z.read(part.get('file'))),BRep_Builder())
                yield obj.get('name'),shape
    def solids(shape):
        ex=TopExp_Explorer(shape,TopAbs_SHELL)
        while ex.More():yield ex.Current();ex.Next()
    def signature(shape):
        v=GProp_GProps();a=GProp_GProps();BRepGProp.VolumeProperties_s(shape,v);BRepGProp.SurfaceProperties_s(shape,a)
        return np.array([abs(v.Mass()),abs(a.Mass())])
    templates={}
    for file,names in [('finger_module.fcstd',{'proximal_01':'proximal','middle_01':'middle','distal_01':'distal'}),('thumb_module.fcstd',{'proximal_thumb_01':'thumb_proximal','distal_thumb_01':'thumb_distal'})]:
        for name,shape in shapes(assembly.parent/file):
            if name in names:
                items=list(solids(shape));assert len(items)==1,(file,name,len(items));templates[names[name]]=signature(items[0])
    records=[];requirements={};counts={};matches={}
    topnames={p['source_id'].rsplit('/',1)[-1] for p in original['parts']}
    finger_names={f'finger_module_0{i}':digit for i,digit in enumerate(['index','middle','ring','little'],1)}
    for name,shape in shapes(assembly):
        if name not in topnames:continue
        components=list(solids(shape))
        def face_count(s):
            ex=TopExp_Explorer(s,TopAbs_FACE);count=0
            while ex.More():count+=1;ex.Next()
            return count
        assert sum(face_count(s) for s in components)==face_count(shape),('face coverage',name)
        for i,solid in enumerate(components):
            v,f=occ_mesh(solid,.001);ident=str(rel)+'#/'+name+f'/shell:{i:03d}'
            file=f'geometry_{len(records):04d}.npz';np.savez_compressed(output/file,vertices=v,faces=f)
            sig=signature(solid);role=None
            for template,t in templates.items():
                if np.allclose(sig,t,rtol=1e-7,atol=1e-5):
                    if name in finger_names and not template.startswith('thumb_'):role=finger_names[name]+'_'+template
                    if name=='thumb_01' and template.startswith('thumb_'):role=template
            if role:requirements[ident]=role;matches[role]=matches.get(role,0)+1
            if name in ('palm_01','dorsal_01'):requirements[ident]='base'
            records.append({'source_id':ident,'source_file':str(rel),'frame':'source_assembly_world','geometry_file':file,'geometry_sha256':sha(output/file),'required':True,'cached_component':name,'matched_phalanx_role':role,**mesh_record(v,f)})
            counts[name]=counts.get(name,0)+1
    expected=['thumb_proximal','thumb_distal']+[d+'_'+p for d in finger_names.values() for p in ['proximal','middle','distal']]
    assert set(matches)==set(expected) and all(n==1 for n in matches.values()),matches
    assert set(counts)==topnames,(counts,topnames)
    inv={**original,'parts':records,'source_coverage_complete':True,'source_body_role_requirements':requirements,'coverage_note':'All BREP shells from all nine visible saved top-level A2plus cached BREP components, without source recomputation. Original cached world placements are frozen in metres. Four fingers each have three uniquely shape-matched phalanges; thumb has two. Rigid-invariant exact BREP volume and area identify template phalanges. This validates saved CAD geometry, not tendon routing, motor transmission, missing original external dependency recomputation, or physical hardware accuracy.','selection_method':'Read original zip-contained BREP; enumerate TopAbs_SHELL; tessellate each original shell independently; preserve cached placement.','cached_component_shell_counts':counts,'unique_phalanx_matches':matches,'original_top_level_inventory_sha256':sha(ROOT/'evaluator/general/references/10_complex/source_inventory.json')}
    dump(output/'source_inventory.json',inv);print('HAND',len(records),counts,matches,flush=True)

def excavator():
    from pxr import Usd
    source=ROOT/'evaluator/general/references/08_excavator';output=HERE/'references/08_excavator'
    shutil.copytree(source,output,dirs_exist_ok=True)
    inv=json.loads((output/'source_inventory.json').read_text());dump(output/'original_native_coverage_inventory.json',inv)
    stage=Usd.Stage.Open(str(output/'reference.usdc'));requirements={};names={}
    mapping={'大臂-1':'boom','小臂-1':'stick','挖斗-1':'bucket','主体底-1':'base','底盘-1':'base'}
    for part in inv['parts']:
        path=part['source_id'].split('#usd:',1)[1];name=stage.GetPrimAtPath(path).GetParent().GetDisplayName();names[part['source_id']]=name
        if name in mapping:requirements[part['source_id']]=mapping[name]
    assert {'base','boom','stick','bucket'}<=set(requirements.values()),requirements
    inv.update(source_coverage_complete=True,source_body_role_requirements=requirements,source_display_names=names,native_assembly_completeness_proven=False,source_coverage_scope='All 80 mesh instances in the frozen independent usd-converter 0.2.0 native SolidWorks conversion.',coverage_note='Canonical reference is the complete 80-mesh converter output, preserving all original native dependency bytes and the converted reference hash. Native SolidWorks assembly completeness and mate fidelity are not independently proven; acceptance is explicitly relative to this frozen neutral conversion. It is not a claim that all original native geometry was converted.')
    inv['canonical_neutral_reference_sha256']=sha(output/'reference.usdc');dump(output/'source_inventory.json',inv);print('EXCAVATOR',len(inv['parts']),requirements,flush=True)

if __name__=='__main__':excavator();hand()
