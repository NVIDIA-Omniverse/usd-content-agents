"""Prepare synthetic controls; native jobs require explicit --run-solver.

Launch through the experiment's global process broker on the qualified remote
host. One native solver child at a time. No benchmark/author USD is an input.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',required=True)
    parser.add_argument('--variant',default='positive',choices=['positive','blocked_motion','no_contact','one_contact','no_friction','kinematic','missing_joint','changed_geometry','gravity_disabled'])
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--run-solver',action='store_true')
    parser.add_argument('--solver-python');parser.add_argument('--seed',type=int,default=11,choices=[11,23,47,83,131]);a=parser.parse_args()
    a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    implementation=a.output/'implementation';implementation.mkdir()
    for p in (ROOT/'cases'/a.case).iterdir():
        if p.is_file() and p.suffix in ('.py','.json') and p.name not in ('frozen_manifest.json','v2_freeze.json'):
            shutil.copy2(p,implementation/p.name)
    shutil.copy2(ROOT/'qualification/fixture_builders'/(a.case+'.py'),implementation/'fixture_builder.py')
    sys.path.insert(0,str(implementation))
    from common import dump,sha,verdict
    from v2_process import run,inspect_with_deadline
    from pxr import Usd,UsdPhysics
    builder=load('fixture_builder',implementation/'fixture_builder.py')
    case=a.case;variant=a.variant;folder=a.output/'generated';config=None;checks=[]
    negative_static=variant in ('kinematic','missing_joint','changed_geometry','gravity_disabled')
    if case=='02_conveyor':
        if variant not in ('positive','blocked_motion','no_contact'):raise ValueError('Use source structural suite for this variant')
        request_path=builder.make(folder,{'blocked_motion':'locked'}.get(variant,variant))
        req=json.loads(request_path.read_text());config=req['config'];scenario=Path(req['scenario']);result_path=folder/'report.json'
        solver_args=[str(implementation/'solver.py'),str(request_path)]
    elif case=='04_gripper':
        if variant not in ('positive','blocked_motion','one_contact','no_friction'):raise ValueError('Use source structural suite for this variant')
        builder.ROOT=folder
        folder,config=builder.fixture({'blocked_motion':'no_motion'}.get(variant,variant))
        scenario=Path(config['scenario']);result_path=folder/'result.json'
        solver_args=[str(implementation/'case04_solver.py'),'--config',str(folder/'config.json'),'--seed',str(a.seed),'--output',str(result_path)]
    else:
        if case in ('08_excavator','10_complex'):
            builder.create(case);folder=implementation/'selftests'/case;source_usd=folder/'positive.usda'
        elif case=='05_vise':
            builder.create_slider(folder);source_usd=folder/'slider.usda'
        else:
            builder.create(folder);source_usd=folder/({'03_hinge':'positive','06_engine':'engine','07_robot_arm':'arm','09_printer':'axes'}[case]+'.usda')
        reference=folder/('slider_reference' if case=='05_vise' else 'reference')
        bindings=json.loads((folder/('slider_bindings.json' if case=='05_vise' else 'bindings.json')).read_text())
        inventory=json.loads((reference/'source_inventory.json').read_text());data=json.loads((implementation/'cases.json').read_text())
        contract=dict(data['cases'][case],common=data['common'],case_id=case)
        stage=Usd.Stage.Open(str(source_usd));first=next(b['path'] for b in bindings['bodies'] if b['moving'])
        if variant=='kinematic':UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(first)).CreateKinematicEnabledAttr(True)
        elif variant=='missing_joint':stage.RemovePrim(bindings['joints'][0]['path'])
        elif variant=='changed_geometry':
            from pxr import UsdGeom
            import numpy as np
            path=next(m['mesh_paths'][0] for m in bindings['source_map'] if m['body_path']==first)
            points=UsdGeom.Mesh(stage.GetPrimAtPath(path)).GetPointsAttr();points.Set((np.asarray(points.Get())*.5).tolist())
        elif variant=='gravity_disabled':
            for prim in stage.Traverse():
                if prim.IsA(UsdPhysics.Scene):UsdPhysics.Scene(prim).CreateGravityMagnitudeAttr(0)
        elif variant=='blocked_motion':
            role=contract['required_joint_roles'][0]
            if case=='09_printer':role='z'
            path=next(j['path'] for j in bindings['joints'] if j['role']==role)
            prim=stage.GetPrimAtPath(path)
            joint=(UsdPhysics.RevoluteJoint if prim.IsA(UsdPhysics.RevoluteJoint) else UsdPhysics.PrismaticJoint)(prim)
            # These branch contracts require a strictly positive limit span.
            # A tiny valid span tests runtime inability to reach the target.
            bound=1e-4 if case in ('08_excavator','10_complex') else 0.
            joint.CreateLowerLimitAttr(-bound);joint.CreateUpperLimitAttr(bound)
        elif variant not in ('positive','no_contact'):raise ValueError('Unsupported control')
        candidate=a.output/'synthetic_input.usda';stage.Flatten().Export(str(candidate));stage=None
        measured=a.output/'measurement';measured.mkdir()
        from structural import inspect
        checks,config=inspect_with_deadline(inspect,candidate,bindings,contract,inventory,folder/'source',reference,measured)
        if case in ('08_excavator','10_complex'):
            from case_hooks import inspect_extra
            checks.extend(inspect_extra(config,bindings))
        if case=='09_printer' and not negative_static:
            from case09_payload import prepare
            ident=next(p['source_id'] for p in inventory['parts'] if p['source_file']=='bed.stl')
            record={'bed_source_ids':[ident],'placement':{'source_center_m':[0,0,.1135],'source_top_normal':[0,0,1]},'source_geometry_hashes_consulted':{ident:next(p['geometry_sha256'] for p in inventory['parts'] if p['source_id']==ident)}}
            assert prepare(config,bindings,inventory,checks,measured,placement_record=record)
            if variant=='no_contact':
                s=Usd.Stage.Open(config['scenario']);s.GetPrimAtPath('/World/bed/Visual').RemoveAPI(UsdPhysics.CollisionAPI);s.GetRootLayer().Save()
        scenario=Path(config['scenario']);result_path=a.output/'result.json'
        solver_args=[str(implementation/'solver.py'),'--config',str(measured/'runtime_config.json'),'--seed',str(a.seed),'--output',str(result_path),'--device','cpu']
    # Small legacy contact fixtures authored their own witness instead of using
    # structural.inspect. Apply the same v2 witness-only instrumentation to it.
    if case in ('02_conveyor','04_gripper'):
        from pxr import Gf,UsdGeom
        s=Usd.Stage.Open(str(scenario));w=s.GetPrimAtPath(config['gravity_witness'])
        UsdGeom.Xformable(w).GetOrderedXformOps()[0].Set(Gf.Vec3d(1,1,1))
        for p in Usd.PrimRange(w):p.RemoveAPI(UsdPhysics.CollisionAPI)
        s.GetRootLayer().Save()
        if case=='02_conveyor':
            req['scene_sha256']=sha(scenario);req['seed']=a.seed;dump(request_path,req)
    dump(a.output/'structural_checks.json',checks)
    structural_failures=[c['name'] for c in checks if not c['passed']]
    expected_prefix={'kinematic':'rigid_body_binding:', 'missing_joint':'required_joint_roles',
                     'changed_geometry':'source_surface_retained:', 'gravity_disabled':'earth_gravity_authored'}.get(variant)
    summary={'case_id':case,'variant':variant,'seed':a.seed,'synthetic_only':True,'source_asset_used':False,
             'native_run':False,'structural_failures':structural_failures,'qualified':any(f.startswith(expected_prefix) for f in structural_failures) if negative_static else not structural_failures,
             'scene_sha256':sha(scenario),'solver_arguments':solver_args,
             'evaluator_code_sha256':{p.name:sha(p) for p in (ROOT/'cases'/case).iterdir() if p.is_file() and p.suffix in ('.py','.json')},
             'fixture_builder_sha256':sha(ROOT/'qualification/fixture_builders'/(case+'.py')),
             'model_calls':0}
    if a.run_solver and not negative_static and not structural_failures:
        if not a.solver_python:raise ValueError('--solver-python is required for explicit native qualification')
        with (a.output/'solver.log').open('w') as log:proc=run([a.solver_python]+solver_args,stdout=log,timeout=900)
        summary['native_run']=True
        if not result_path.is_file():summary.update(qualified=False,disposition='INCONCLUSIVE',reason='Native result missing')
        else:
            result=json.loads(result_path.read_text())
            if case=='02_conveyor':
                accepted=result.get('status')=='PASS';concrete=[k for k,v in result.get('checks',{}).items() if not v and k!='independent_gravity'];report=result
            else:
                if case=='04_gripper':from case04_evaluate import observe
                else:from evaluate import observe
                records=observe(result,config)
                if case=='09_printer':
                    from case09_payload import observe as payload_observe
                    records.extend(payload_observe(result,config))
                report=verdict(records);accepted=report['accepted'];concrete=report['concrete_failures']
            summary.update(qualified=accepted if variant=='positive' else bool(concrete),native_accepted=accepted,concrete_failures=concrete,result_sha256=sha(result_path))
            dump(a.output/'observed.json',report)
            if case=='06_engine' and variant=='positive' and a.seed==11 and accepted:
                baseline=dict(config,external_load_scale=0.0);baseline_path=a.output/'unloaded_config.json';dump(baseline_path,baseline)
                unloaded_path=a.output/'unloaded_result.json'
                with (a.output/'unloaded_solver.log').open('w') as log:
                    run([a.solver_python,str(implementation/'solver.py'),'--config',str(baseline_path),'--seed','11','--output',str(unloaded_path),'--device','cpu'],stdout=log,timeout=900)
                if unloaded_path.is_file():
                    unloaded=json.loads(unloaded_path.read_text());joint=config['joint_roles']['crank']
                    differences=[abs(x['joints'][joint]['q_unwrapped']-y['joints'][joint]['q_unwrapped']) for x,y in zip(result.get('trace',[]),unloaded.get('trace',[])) if 5.6<=x['t']<=6.5]
                    summary['paired_unloaded']={'max_angle_change_rad':max(differences) if differences else None,'required_rad':1e-4,'passed':bool(differences) and max(differences)>=1e-4,'result_sha256':sha(unloaded_path)}
                else:summary['paired_unloaded']={'passed':False,'disposition':'INCONCLUSIVE','reason':'Missing unloaded native output'}
                summary['qualified'] &= summary['paired_unloaded']['passed']
    elif not negative_static:
        summary.update(qualified=False,disposition='NATIVE_QUALIFICATION_PENDING',preparation_passed=not structural_failures)
    dump(a.output/'qualification.json',summary);print(json.dumps(summary,indent=2))
    raise SystemExit(0 if summary['qualified'] or summary.get('preparation_passed') else 1)


if __name__=='__main__':main()
