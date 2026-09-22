"""Synthetic boxes only, for evaluator qualification. No benchmark solution."""
import json,copy
from pathlib import Path
import numpy as np
import trimesh
from pxr import Gf,Usd,UsdGeom,UsdPhysics
from common import dump
from geometry import freeze

HERE=Path(__file__).resolve().parent

def create(case):
    c=json.loads((HERE/'cases.json').read_text())['cases'][case]
    root=HERE/'selftests'/case;root.mkdir(parents=True,exist_ok=True);source=root/'source';source.mkdir(exist_ok=True)
    positions={'base':np.array([-.04,0.,0.])}
    if case=='08_excavator':
        for i,role in enumerate(['boom','stick','bucket']):positions[role]=np.array([i*.05,0.,0.])
    else:
        for d,digit in enumerate(['thumb','index','middle','ring','little']):
            for i,part in enumerate(['proximal','distal'] if digit=='thumb' else ['proximal','middle','distal']):positions[digit+'_'+part]=np.array([i*.05,(d-2)*.15,0.])
    meshes={}
    for role in c['required_body_roles']:
        mesh=trimesh.creation.box(extents=[.02,.7,.02] if role=='base' else [.04,.006,.006]);mesh.apply_translation([.02,0,0] if role!='base' else [0,0,0]);mesh.export(source/(role+'.stl'));meshes[role]=mesh
    inv=freeze(source,[Path(r+'.stl') for r in meshes],1,root/'reference',case)
    inv['source_body_role_requirements']={p['source_id']:Path(p['source_file']).stem for p in inv['parts']};dump(root/'reference/source_inventory.json',inv)
    stage=Usd.Stage.CreateNew(str(root/'positive.usda'));UsdGeom.SetStageMetersPerUnit(stage,1);UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
    world=UsdGeom.Xform.Define(stage,'/World');stage.SetDefaultPrim(world.GetPrim())
    scene=UsdPhysics.Scene.Define(stage,'/World/Physics');scene.CreateGravityDirectionAttr(Gf.Vec3f(0,0,-1));scene.CreateGravityMagnitudeAttr(9.81)
    bindings={'schema_version':1,'case_id':case,'bodies':[],'joints':[],'source_map':[],'assembly_from_source':np.eye(4).tolist()}
    for role,mesh in meshes.items():
        body=UsdGeom.Xform.Define(stage,'/World/'+role);body.AddTranslateOp().Set(Gf.Vec3d(*positions[role]));path=str(body.GetPath());m=UsdGeom.Mesh.Define(stage,path+'/Visual')
        m.CreatePointsAttr(mesh.vertices.tolist());m.CreateFaceVertexCountsAttr([3]*len(mesh.faces));m.CreateFaceVertexIndicesAttr(mesh.faces.reshape(-1).tolist());m.CreateSubdivisionSchemeAttr('none')
        UsdPhysics.CollisionAPI.Apply(m.GetPrim());UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr('convexHull')
        if role!='base':
            UsdPhysics.RigidBodyAPI.Apply(body.GetPrim());mass=UsdPhysics.MassAPI.Apply(body.GetPrim());mass.CreateMassAttr(.02);mass.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-5,1e-5,1e-5));mass.CreateCenterOfMassAttr(Gf.Vec3f(.02,0,0))
        bindings['bodies'].append({'path':path,'role':role,'moving':role!='base'})
        bindings['source_map'].append({'source_id':next(p['source_id'] for p in inv['parts'] if p['source_file']==role+'.stl'),'mesh_paths':[str(m.GetPath())],'body_path':path,'source_to_body':np.eye(4).tolist()})
    for role,(a,b) in c['required_joint_body_pairs'].items():
        path='/World/joint_'+role;j=UsdPhysics.RevoluteJoint.Define(stage,path);j.CreateBody0Rel().SetTargets(['/World/'+a]);j.CreateBody1Rel().SetTargets(['/World/'+b]);j.CreateAxisAttr('Z');j.CreateLowerLimitAttr(-10);j.CreateUpperLimitAttr(45)
        j.CreateLocalPos0Attr(Gf.Vec3f(*(positions[b]-positions[a])));j.CreateLocalPos1Attr(Gf.Vec3f(0));bindings['joints'].append({'path':path,'role':role})
    stage.GetRootLayer().Save();dump(root/'bindings.json',bindings)
    for name in ['kinematic','changed_geometry','wrong_branch','missing_joint']:
        copy_stage=Usd.Stage.Open(str(root/'positive.usda'));copy_stage.GetRootLayer().Reload();first=c['required_body_roles'][1];joint_role=c['required_joint_roles'][-1]
        if name=='kinematic':UsdPhysics.RigidBodyAPI(copy_stage.GetPrimAtPath('/World/'+first)).CreateKinematicEnabledAttr(True)
        if name=='changed_geometry':UsdGeom.Mesh(copy_stage.GetPrimAtPath('/World/'+first+'/Visual')).GetPointsAttr().Set((np.asarray(meshes[first].vertices)*.5).tolist())
        if name=='wrong_branch':UsdPhysics.Joint(copy_stage.GetPrimAtPath('/World/joint_'+joint_role)).GetBody0Rel().SetTargets(['/World/base'])
        if name=='missing_joint':copy_stage.RemovePrim('/World/joint_'+joint_role)
        copy_stage.Flatten().Export(str(root/(name+'.usda')))
    print(root,flush=True)

if __name__=='__main__':
    for case in ['08_excavator','10_complex']:create(case)
