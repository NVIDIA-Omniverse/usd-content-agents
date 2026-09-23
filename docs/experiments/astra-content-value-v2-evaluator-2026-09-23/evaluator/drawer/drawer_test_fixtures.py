"""Synthetic evaluator tests only. Does not author the Poly Haven cabinet."""
import copy, json, os, pathlib, subprocess, sys
from pxr import Gf, Usd, UsdGeom, UsdPhysics
import drawer_evaluate as E
ROOT=pathlib.Path(os.environ.get('DRAWER_QUALIFICATION_DIR',str(E.HERE/'selftest')));ROOT.mkdir(parents=True,exist_ok=True)

def cube(s,path,center,size,collision=True):
    p=UsdGeom.Cube.Define(s,path);p.CreateSizeAttr(1);p.AddTranslateOp().Set(Gf.Vec3d(*center));p.AddScaleOp().Set(Gf.Vec3f(*size))
    if collision:UsdPhysics.CollisionAPI.Apply(p.GetPrim())
    return p.GetPrim()

def fixture(name,variant='good'):
    path=ROOT/(name+'.usda');s=Usd.Stage.CreateNew(str(path));UsdGeom.SetStageUpAxis(s,'Y');UsdGeom.SetStageMetersPerUnit(s,1)
    UsdGeom.Xform.Define(s,'/World');cab=UsdGeom.Xform.Define(s,'/World/Cabinet');cube(s,'/World/Cabinet/TinyAnchor',[2,0,0],[.01,.01,.01])
    body=UsdGeom.Xform.Define(s,'/World/Slider').GetPrim();rb=UsdPhysics.RigidBodyAPI.Apply(body);rb.CreateKinematicEnabledAttr(variant=='kinematic');ma=UsdPhysics.MassAPI.Apply(body);ma.CreateMassAttr(2);ma.CreateDiagonalInertiaAttr(Gf.Vec3f(.03,.2,.2));ma.CreateCenterOfMassAttr(Gf.Vec3f(0,.98,0))
    coll=variant!='no_colliders'
    cube(s,'/World/Slider/Floor',[0,.96,0],[1,.01,.36],coll)
    cube(s,'/World/Slider/Left',[-.5,1.015,0],[.02,.12,.38],coll);cube(s,'/World/Slider/Right',[.5,1.015,0],[.02,.12,.38],coll)
    cube(s,'/World/Slider/Back',[0,1.015,-.18],[1,.12,.02],coll);cube(s,'/World/Slider/Front',[0,1.015,.18],[1,.12,.02],coll)
    if variant=='filled':cube(s,'/World/Slider/BadFilledVolume',[0,1.03,0],[.96,.1,.34])
    j=(UsdPhysics.RevoluteJoint if variant=='wrong_joint' else UsdPhysics.PrismaticJoint).Define(s,'/World/SlideJoint');j.CreateBody1Rel().SetTargets(['/World/Slider']);j.CreateAxisAttr('Z');j.CreateLowerLimitAttr(0);j.CreateUpperLimitAttr(.3)
    if variant=='no_motion':
        fixed=UsdPhysics.FixedJoint.Define(s,'/World/Blocked');fixed.CreateBody1Rel().SetTargets(['/World/Slider'])
    s.GetRootLayer().Save()
    b={'asset_id':'synthetic_fixture','_usd_path':str(path),'drawer_body':'/World/Slider','cabinet_body':'/World/Cabinet','drawer_joint':'/World/SlideJoint'}
    return s,b

def run():
    results=[]
    for variant,needle in [('kinematic','kinematic'),('no_colliders','no enabled colliders'),('wrong_joint','prismatic')]:
        s,b=fixture(variant,variant);errors,_=E.validate_core(s,b);ok=any(needle in x for x in errors);results.append({'test':variant,'pass':ok,'errors':errors})
    s,b=fixture('synthetic_good');errors,_=E.validate_core(s,b);results.append({'test':'synthetic_structural_positive','pass':not errors,'errors':errors})
    ref=E.gltf_reference();a=ref['drawer_cabinet_drawer_01'];bad=a.copy();bad[:,:,2]*=.75
    results.append({'test':'source_fidelity_positive','pass':E.fidelity(a,a)['pass']})
    results.append({'test':'source_fidelity_shape_change_negative','pass':not E.fidelity(a,bad)['pass']})
    for name in ['good','no_motion','filled']:
        s,b=fixture('dynamic_'+name,name);out=ROOT/('run_'+name);out.mkdir(exist_ok=True)
        if name in ['good','filled']:
            preflight=E.collision_preflight(s,b,out,os.environ['OVPHYSX_TEST_PYTHON'])
            expected_query='PASS' if name=='good' else 'FAIL'
            results.append({'test':'initialized_query_'+name,'pass':preflight['status']==expected_query,'expected':expected_query,'actual':preflight})
        req=E.trial_scene(s,b,11,out)
        with (out/'solver.log').open('w') as log:
            r=subprocess.run([os.environ['OVPHYSX_TEST_PYTHON'],str(E.HERE/'drawer_solver.py'),str(out/'request.json')],stdout=log,stderr=subprocess.STDOUT,timeout=300)
        report=json.loads((out/'trial_report.json').read_text());expected='PASS' if name=='good' else 'FAIL'
        ok=report['status']==expected
        if name=='no_motion':ok=ok and not report.get('checks',{}).get('opening',True)
        if name=='filled':ok=ok and not report.get('checks',{}).get('interior_clear',True)
        results.append({'test':'native_'+name,'pass':ok,'expected':expected,'actual':report})
        if name=='good' and (out/'trace.jsonl').exists():E.replay(out/'scene.usda',out/'trace.jsonl',out/'replay.usda',[b['drawer_body'],'/__EvaluatorPayload'])
    request=json.loads((ROOT/'run_good/initial_collision_preflight/request.json').read_text())
    request['witness_center']=[100.0,100.0,100.0];request['output']=str(ROOT/'missing_query_witness_report.json')
    E.write_json(ROOT/'missing_query_witness_request.json',request)
    with (ROOT/'missing_query_witness.log').open('w') as log:
        subprocess.run([os.environ['OVPHYSX_TEST_PYTHON'],str(E.HERE/'drawer_collision_preflight.py'),str(ROOT/'missing_query_witness_request.json')],stdout=log,stderr=subprocess.STDOUT,timeout=300)
    missing=json.loads((ROOT/'missing_query_witness_report.json').read_text())
    results.append({'test':'absent_query_witness_cannot_pass','pass':missing['status']=='INCONCLUSIVE' and not missing['pass'],'actual':missing})
    result={'scope':'synthetic evaluator fixtures only, not a scored source cabinet trial','pass':all(x['pass'] for x in results),'tests':results};E.write_json(ROOT/'test_report.json',result);print(json.dumps(result,indent=2));return result
if __name__=='__main__':sys.exit(0 if run()['pass'] else 1)
