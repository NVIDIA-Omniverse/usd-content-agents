"""Positive/negative USD fixture qualification, entirely synthetic and CPU-only."""
import argparse, copy, json, hashlib, platform
from pathlib import Path
from pxr import Usd, UsdGeom, UsdPhysics
from audit import observe,sha,write,stamp

PUBLIC={'required_body_roles':['base','moving'],'required_joint_roles':['hinge'],'joint_kinds':{'hinge':'revolute'}}
BINDINGS={'bodies':[{'role':'base','path':'/World/Base'},{'role':'moving','path':'/World/Moving','moving':True}],
          'joints':[{'role':'hinge','path':'/World/Hinge'}]}

def fixture(path, mode):
    s=Usd.Stage.CreateNew(str(path));UsdGeom.Xform.Define(s,'/World');UsdGeom.Xform.Define(s,'/World/Base')
    mover=UsdGeom.Xform.Define(s,'/World/Moving').GetPrim();shape=UsdGeom.Cube.Define(s,'/World/Moving/Shape').GetPrim()
    if mode!='geometry_only':
        rb=UsdPhysics.RigidBodyAPI.Apply(mover)
        rb.CreateRigidBodyEnabledAttr(mode!='disabled_body');rb.CreateKinematicEnabledAttr(mode=='kinematic_body')
        if mode!='no_collider':UsdPhysics.CollisionAPI.Apply(shape).CreateCollisionEnabledAttr(mode!='disabled_collider')
        joint=(UsdPhysics.PrismaticJoint if mode=='wrong_joint_kind' else UsdPhysics.RevoluteJoint).Define(s,'/World/Hinge')
        joint.CreateBody0Rel().SetTargets(['/World/Base']);joint.CreateBody1Rel().SetTargets(['/World/Moving'])
        joint.CreateJointEnabledAttr(mode!='disabled_joint')
    s.GetRootLayer().Save()

def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();assert not a.output_dir.exists()
    a.output_dir.mkdir(parents=True);tests=[]
    for mode in ['positive','geometry_only','disabled_body','kinematic_body','no_collider','disabled_collider','wrong_joint_kind','disabled_joint','missing_body_binding','missing_joint_binding','fake_body_path','fake_joint_path','duplicate_body_role','duplicate_joint_role']:
        path=a.output_dir/(mode+'.usda');fixture(path,mode);b=copy.deepcopy(BINDINGS)
        if mode=='missing_body_binding':b['bodies']=b['bodies'][:1]
        if mode=='missing_joint_binding':b['joints']=[]
        if mode=='fake_body_path':b['bodies'][1]['path']='/World/Missing'
        if mode=='fake_joint_path':b['joints'][0]['path']='/World/Moving'
        if mode=='duplicate_body_role':b['bodies'].append(dict(b['bodies'][1]))
        if mode=='duplicate_joint_role':b['joints'].append(dict(b['joints'][0]))
        before=sha(path);result=observe(path,b,PUBLIC)
        expected=mode!='positive';actual=result['supplementary_status']=='DECISIVE_NON_DELIVERY'
        tests.append({'name':mode,'expected_non_delivery':expected,'actual_non_delivery':actual,'passed':actual==expected and sha(path)==before,'scene_sha256':before,'observation':result})
    # Missing composed dependencies must raise and cannot be mistaken for proof
    # of a submitted absence. This fixture is deliberately invalid.
    missing=a.output_dir/'unresolved_dependency.usda';fixture(missing,'positive');s=Usd.Stage.Open(str(missing));s.GetRootLayer().subLayerPaths=['missing_layer.usda'];s.GetRootLayer().Save()
    try:observe(missing,BINDINGS,PUBLIC)
    except ValueError as e:blocked=True;message=str(e)
    else:blocked=False;message='unexpected_success'
    tests.append({'name':'unresolved_dependency_not_decisive','passed':blocked,'diagnostic':message})
    result={'created_utc':stamp(),'synthetic_only':True,'all_passed':all(x['passed'] for x in tests),'test_count':len(tests),'tests':tests,
            'auditor_sha256':sha(Path(__file__).with_name('audit.py')),'qualifier_sha256':sha(__file__),
            'usd_version':list(Usd.GetVersion()),'python_version':platform.python_version(),'gpu_jobs':0,'proximity_queries':0,
            'scope':'Qualifies static necessary-role/API presence observations only. Positive fixture is never called a simulated or source-faithful acceptance.'}
    write(a.output_dir/'qualification.json',result);print(json.dumps({'all_passed':result['all_passed'],'tests':len(tests)}));assert result['all_passed']
if __name__=='__main__':main()
