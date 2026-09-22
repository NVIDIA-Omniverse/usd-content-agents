"""Versioned post-freeze geometry adjudication. Never edits frozen or author files.

The only numerical change is a uniform x1000 conversion of both meshes and
length tolerance before calling the frozen surface_compare function, followed
by conversion of measured lengths back to meters. Native physics is reused
only with complete five-seed evidence and unchanged hashed inputs/config.
The conveyor may need a fresh run of its unchanged frozen solver because v1
gated all physics on geometry. That separate run is explicitly recorded.
"""
import argparse, ast, contextlib, copy, datetime, hashlib, importlib.util
import json, os, subprocess, sys, time, traceback
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import trimesh

VERSION='source_distance_v1_1'
SEEDS=[11,23,47,83,131]
CASES=['02_conveyor','03_hinge','04_gripper','05_vise','06_engine','07_robot_arm','08_excavator','09_printer','10_complex']
SURFACE_SOURCE_SHA='e402e7a4511ead7db0ef45f28c7de7ac8c58a1ec59279ddbce7abd827e2c99dd'
HERE=Path(__file__).resolve().parent

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def load(path):return json.loads(Path(path).read_text())
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def dump(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    assert not path.exists(), 'Refusing to overwrite evidence: '+str(path)
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def record(name,passed,evidence=None,insufficient=False):
    return {'name':name,'passed':bool(passed),'evidence':evidence,'failure_class':None if passed else ('evaluator_insufficient' if insufficient else 'rejected')}
def verdict(checks):
    failures=[x for x in checks if not x['passed']]
    bad=[x['name'] for x in failures if x['failure_class']!='evaluator_insufficient']
    inc=[x['name'] for x in failures if x['failure_class']=='evaluator_insufficient']
    return {'accepted':not failures,'status':'accepted' if not failures else ('not_accepted' if bad else 'inconclusive'),'concrete_failures':bad,'inconclusive_checks':inc,'checks':checks}

def setup(snapshot):
    code=(snapshot/'geometry.py').read_text()
    node=next(x for x in ast.parse(code).body if isinstance(x,ast.FunctionDef) and x.name=='surface_compare')
    assert hashlib.sha256(ast.get_source_segment(code,node).strip().encode()).hexdigest()==SURFACE_SOURCE_SHA,'Unqualified surface algorithm version'
    sys.path.insert(0,str(snapshot))
    import geometry, structural
    return geometry.surface_compare,structural

def corrected(raw,sv,sf,fv,ff,tol):
    passed,d=raw(np.asarray(sv)*1000,sf,np.asarray(fv)*1000,ff,float(tol)*1000)
    for k in ['source_to_final_max_sample_distance_m','final_to_source_max_sample_distance_m','bounds_error_m','tolerance_m']:d[k]/=1000
    d['method']+='; versioned uniform x1000 length-unit evaluation, reported lengths converted back to meters'
    return passed,d

def qualify(args):
    raw,_=setup(args.snapshot);rows=[]
    for edge in [.0001,.0002,.0005,.001,.01]:
        for origin in [np.zeros(3),np.array([0,.1,.065]),np.array([1.2,-2.3,3.4])]:
            v=np.array([[0,0,0],[edge,0,0],[0,edge,0]])+origin;f=np.array([[0,1,2]])
            mesh=trimesh.Trimesh(v*1000,f,process=False)
            for offset in [0,.000025,.000075]:
                p=np.array([[edge/4,edge/4,offset]])+origin
                _,distance,_=trimesh.proximity.closest_point(mesh,p*1000);measured=float(distance[0]/1000)
                rows.append({'name':'analytic_point_distance','edge_m':edge,'origin_m':origin.tolist(),'expected_m':offset,'measured_m':measured,'passed':abs(measured-offset)<1e-12})
    v=np.array([[0,0,0],[.0005,0,0],[0,.0005,0]])+np.array([0,.1,.065]);f=np.array([[0,1,2]])
    for name,final,want in [('identity',v,True),('25um_offset_below50um',v+[0,0,.000025],True),('75um_offset_above50um',v+[0,0,.000075],False),('quadruple_area',(v-[0,.1,.065])*2+[0,.1,.065],False)]:
        accepted,detail=corrected(raw,v,f,final,f,.00005)
        a,b=trimesh.Trimesh(v,f,process=False),trimesh.Trimesh(final,f,process=False)
        ratio=float(b.area/a.area);bounds=float(np.max(np.abs(a.bounds-b.bounds)))
        rows.append({'name':name,'expected_accepted':want,'actual_accepted':accepted,'detail':detail,'meter_area_ratio':ratio,'meter_bounds_error_m':bounds,'passed':accepted==want and abs(ratio-detail['area_ratio'])<1e-10 and abs(bounds-detail['bounds_error_m'])<1e-12})
    old,detail=raw(v,f,v,f,.00005)
    rows.append({'name':'frozen_identity_defect_is_reproduced','expected_accepted':False,'actual_accepted':old,'detail':detail,'passed':not old})
    report={'version':VERSION,'qualified_utc':utc(),'scope':'Unscored synthetic regressions only','all_passed':all(x['passed'] for x in rows),'tests':rows,'worker_sha256':sha(__file__),'surface_compare_source_sha256':SURFACE_SOURCE_SHA,'frozen_geometry_sha256':sha(args.snapshot/'geometry.py'),'trimesh_version':trimesh.__version__,'numpy_version':np.__version__,'length_scale_factor':1000,'tolerance_changes':False,'applies_to_cases':CASES}
    dump(args.qualification,report);assert report['all_passed'],'Qualification failed'
    print(json.dumps({'qualification':str(args.qualification),'all_passed':True,'tests':len(rows),'sha256':sha(args.qualification)}),flush=True)

def discover(root,case,run_id):
    if case=='02_conveyor':snapshot=root/'evaluator/conveyor';inventory=snapshot/'source_inventory.json'
    else:
        family='case08_10' if case in ['08_excavator','10_complex'] else 'general'
        snapshot=root/'evaluator'/family/'frozen'/case;inventory=root/'evaluator'/family/'references'/case/'source_inventory.json'
    if not snapshot.exists():
        candidates=[root/'evaluations'/run_id/case/'_runtime/frozen'/case,root/'evaluations'/run_id/case/'_runtime'/case]
        snapshot=next((p for p in candidates if p.is_dir()),snapshot)
    if not inventory.exists():
        candidates=[root/'evaluations'/run_id/case/'_runtime/references'/case/'source_inventory.json']
        inventory=next((p for p in candidates if p.is_file()),inventory)
    return snapshot,inventory

def physics_evidence(folder,case):
    records=[];files={}
    config=folder/'runtime_config.json'
    if config.is_file():
        files[str(config)]=sha(config)
        scenario=Path(load(config).get('scenario',''))
        if scenario.is_file():files[str(scenario)]=sha(scenario)
    for seed in SEEDS:
        path=folder/(f'seed_{seed}/report.json' if case=='02_conveyor' else f'seed_{seed}.json')
        if not path.is_file():records.append({'seed':seed,'complete':False,'missing':str(path)});continue
        d=load(path);files[str(path)]=sha(path)
        complete=(d.get('checks',{}).get('completed') is True and d.get('status')!='INCONCLUSIVE') if case=='02_conveyor' else (d.get('completed_steps')==d.get('expected_steps') and isinstance(d.get('expected_steps'),int) and d['expected_steps']>0 and 'infrastructure_error' not in d)
        records.append({'seed':seed,'complete':bool(complete),'path':str(path),'sha256':files[str(path)]})
    return {'all_five_complete':len(records)==5 and all(x['complete'] for x in records),'seeds':records,'configuration_and_result_hashes':files}

def wait_ready(paths,seconds):
    deadline=time.monotonic()+seconds
    while not all(p.is_file() for p in paths):
        if time.monotonic()>=deadline:raise TimeoutError('Required completed v1/author evidence absent: '+', '.join(str(p) for p in paths if not p.is_file()))
        time.sleep(min(10,max(.1,deadline-time.monotonic())))

def adjudicate(args):
    started=utc();run=args.root/'runs'/args.run_id/args.case/args.arm
    v1dir=args.v1_output or args.root/'evaluations'/args.run_id/args.case/args.arm
    out=args.output or args.root/'evaluation_adjudications'/args.run_id/args.case/args.arm/VERSION
    wait_ready([run/'execution.json',run/'output_manifest.json',v1dir/'acceptance.json'],args.wait_seconds)
    assert not out.exists(),'Refusing to overwrite adjudication'
    q=load(args.qualification);assert q['all_passed'] and q['worker_sha256']==sha(__file__) and q['surface_compare_source_sha256']==SURFACE_SOURCE_SHA and q['trimesh_version']==trimesh.__version__
    snapshot,inventory=discover(args.root,args.case,args.run_id);snapshot=args.snapshot or snapshot;inventory=args.inventory or inventory
    freeze_path=snapshot/('freeze.json' if args.case=='02_conveyor' else 'frozen_manifest.json');frozen=load(freeze_path)
    code=frozen.get('code_sha256',frozen.get('files'));assert code and all(sha(snapshot/n)==h for n,h in code.items())
    assert sha(inventory)==frozen.get('reference_inventory_sha256',frozen.get('source_inventory_sha256'))
    raw,structural=setup(snapshot)
    from pxr import Usd
    source=load(inventory);bindings_path=None;submission=load(run/'submission.json');manifest={x['path']:x for x in load(run/'output_manifest.json')['files']};inputs={}
    for key in ['final_scene','bindings']:
        relative=Path(submission[key]);p=(run/relative).resolve()
        assert not relative.is_absolute() and p.is_relative_to(run.resolve()) and p.is_file()
        assert sha(p)==manifest[str(relative)]['sha256'];inputs[key]=p
    v1=load(v1dir/'acceptance.json')
    assert v1.get('input_sha256',v1.get('scene_sha256'))==sha(inputs['final_scene']) and v1['bindings_sha256']==sha(inputs['bindings'])
    expected_inventory=v1.get('inventory_sha256',v1.get('source_inventory_sha256'));assert expected_inventory==sha(inventory)
    source_root=args.source_root or args.root/'assets'/args.case
    for f in source['source_files']:assert sha(source_root/f['path'])==f['sha256']
    out.mkdir(parents=True)
    before_physics=physics_evidence(v1dir,args.case)
    old_checks={c['name']:c for c in v1['checks']};newchecks={};rows=[]
    bindings=load(inputs['bindings']);stage=Usd.Stage.Open(str(inputs['final_scene']),load=Usd.Stage.LoadAll)
    mapped={x['source_id']:x for x in bindings['source_map']};assembly=np.asarray(bindings['assembly_from_source'],float)
    common=load(snapshot/'cases.json')['common'] if args.case!='02_conveyor' else {'max_mesh_surface_error_m':.00005,'max_mesh_surface_error_fraction':.001}
    for part in source['parts']:
        if not part['required']:continue
        ident=part['source_id'];name='source_surface_retained:'+ident
        if name not in old_checks:continue
        try:
            mapping=mapped[ident];npz=inventory.parent/part['geometry_file'];assert sha(npz)==part['geometry_sha256'];ref=np.load(npz)
            v,f=structural.combine(stage,mapping['mesh_paths'])
            matrix=assembly if part['frame']=='source_assembly_world' else structural.world_matrix(stage.GetPrimAtPath(mapping['body_path']))@np.asarray(mapping['source_to_body'])
            expected=ref['vertices']@matrix[:3,:3].T+matrix[:3,3]
            tol=max(common['max_mesh_surface_error_m'],float(np.linalg.norm(np.ptp(expected,axis=0)))*common['max_mesh_surface_error_fraction'])
            prior=old_checks[name]['evidence'];assert abs(tol-prior['tolerance_m'])<1e-12
            okay,detail=corrected(raw,expected,ref['faces'],v,f,tol)
            a,b=trimesh.Trimesh(expected,ref['faces'],process=False),trimesh.Trimesh(v,f,process=False)
            area=float(b.area/a.area);bounds=float(np.max(np.abs(a.bounds-b.bounds)))
            independent_identical=abs(area-detail['area_ratio'])<1e-10 and abs(bounds-detail['bounds_error_m'])<1e-12 and abs(area-prior['area_ratio'])<1e-10 and abs(bounds-prior['bounds_error_m'])<1e-12
            assert independent_identical,'Non-distance quantity changed'
            detail.update(area_ratio=prior['area_ratio'],bounds_error_m=prior['bounds_error_m'],tolerance_m=prior['tolerance_m'],non_distance_values_preserved_exactly_from_v1=True)
            newchecks[name]=record(name,okay,detail)
            rows.append({'source_id':ident,'v1_passed':old_checks[name]['passed'],'v1_1_passed':okay,'v1_distance_m':max(prior['source_to_final_max_sample_distance_m'],prior['final_to_source_max_sample_distance_m']),'v1_1_distance_m':max(detail['source_to_final_max_sample_distance_m'],detail['final_to_source_max_sample_distance_m']),'tolerance_m':tol,'bounds_pass':bounds<=tol,'area_pass':abs(area-1)<=.03,'independent_meter_bounds_error_m':bounds,'independent_meter_area_ratio':area,'non_distance_values_identical':independent_identical})
        except Exception as exc:
            newchecks[name]=record(name,False,{'diagnostic_error':repr(exc)},True);rows.append({'source_id':ident,'v1_1_passed':False,'inconclusive':repr(exc)})
        print(json.dumps({'case':args.case,'arm':args.arm,'part':len(rows),'source_id':ident,'passed':newchecks[name]['passed']}),flush=True)
    checks=[newchecks.get(x['name'],copy.deepcopy(x)) for x in v1['checks']]
    conservative=[]
    for x in v1['checks']:
        if x['name'].startswith('source_surface_retained:'):
            e=x['evidence'];non_distance_fail=e['bounds_error_m']>e['tolerance_m'] or abs(e['area_ratio']-1)>.03
            conservative.append(record(x['name'],False,{'frozen_v1_evidence':e,'erratum':'Meter closest-point distance is numerically unqualified for tiny triangles; only bounds/area remain independent.'},not non_distance_fail))
        else:conservative.append(copy.deepcopy(x))
    fresh=None;physics=before_physics;physics_dir=v1dir
    if args.case=='02_conveyor' and args.fresh_conveyor_if_needed and not before_physics['all_five_complete'] and checks and all(x['passed'] for x in checks):
        structural.surface_compare=lambda sv,sf,fv,ff,t:corrected(raw,sv,sf,fv,ff,t)
        spec=importlib.util.spec_from_file_location('frozen_conveyor_evaluator',snapshot/'evaluate.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        physical_out=out/'fresh_native'
        options=SimpleNamespace(usd=inputs['final_scene'],bindings=inputs['bindings'],inventory=inventory,source_root=str(source_root),output=physical_out,solver_python=str(args.solver_python or args.root/'ovphysx-venv/bin/python'))
        with (out/'fresh_native_evaluator.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):fresh=module.evaluate(options)
        checks=fresh['checks'];physics=physics_evidence(physical_out,args.case);physics_dir=physical_out
    if not physics['all_five_complete']:
        checks.append(record('v1_1_complete_native_physics_required',False,'Five complete independent native seed results are required; geometry correction alone never establishes acceptance.',True))
    assert all(sha(Path(p))==h for p,h in before_physics['configuration_and_result_hashes'].items()),'v1 config/native evidence changed during adjudication'
    assert all(sha(inputs[k])==manifest[str(inputs[k].relative_to(run))]['sha256'] for k in inputs)
    assert all(sha(snapshot/n)==h for n,h in code.items()),'Frozen code changed during adjudication'
    result=verdict(checks);conservative_result=verdict(conservative)
    base={'version':VERSION,'case_id':args.case,'arm':args.arm,'started_utc':started,'finished_utc':utc(),'worker_sha256':sha(__file__),'qualification_sha256':sha(args.qualification),'frozen_snapshot_manifest_sha256':sha(freeze_path),'frozen_code_sha256':code,'v1_acceptance_sha256':sha(v1dir/'acceptance.json'),'input_scene_sha256':sha(inputs['final_scene']),'bindings_sha256':sha(inputs['bindings']),'source_inventory_sha256':sha(inventory),'tolerances_changed':False,'frozen_files_modified':False,'scored_artifacts_modified':False,'uniform_length_scale':1000,'source_surface_checks_replaced':len(newchecks),'geometry_parts':rows,'native_physics':physics,'native_physics_origin':'fresh_unchanged_frozen_conveyor_solver' if fresh else 'unchanged_frozen_v1_results','native_physics_directory':str(physics_dir),'v1_native_config_and_results_unchanged':True,'claimed_accepted':submission.get('claimed_accepted')}
    result.update(base);result['false_pass']=bool(submission.get('claimed_accepted') is True and result['concrete_failures'])
    conservative_result.update(version='v1_conservative_after_distance_erratum',case_id=args.case,arm=args.arm,v1_acceptance_sha256=base['v1_acceptance_sha256'],false_pass=bool(submission.get('claimed_accepted') is True and conservative_result['concrete_failures']))
    dump(out/'conservative_v1.json',conservative_result);dump(out/'acceptance.json',result)
    dump(out/'provenance.json',{'version':VERSION,'hashes':{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()},'author_manifest_sha256':sha(run/'output_manifest.json'),'v1_physics_before':before_physics,'worker_sha256':sha(__file__)})
    print(json.dumps({'case':args.case,'arm':args.arm,'status':result['status'],'accepted':result['accepted'],'false_pass':result['false_pass'],'geometry_passed':sum(x['v1_1_passed'] for x in rows),'geometry_checked':len(rows),'all_five_native_complete':physics['all_five_complete'],'output':str(out)}),flush=True)

def watch_all(args):
    assert not any([args.snapshot,args.inventory,args.v1_output,args.output]),'Batch mode discovers standard paths; use single-job overrides for staged runtimes'
    cases=CASES if args.case=='all' else [args.case];arms=['plain_astra','content_agents'] if args.arm=='both' else [args.arm]
    pending={(c,a) for c in cases for a in arms};deadline=time.monotonic()+args.wait_seconds
    while pending:
        progress=False
        for case,arm in sorted(pending):
            run=args.root/'runs'/args.run_id/case/arm;v1=args.root/'evaluations'/args.run_id/case/arm/'acceptance.json';done=args.root/'evaluation_adjudications'/args.run_id/case/arm/VERSION/'acceptance.json'
            if done.is_file():pending.remove((case,arm));progress=True;continue
            if not all(x.is_file() for x in [run/'execution.json',run/'output_manifest.json',v1]):continue
            snapshot,inventory=discover(args.root,case,args.run_id)
            if not snapshot.is_dir() or not inventory.is_file():continue
            command=[sys.executable,str(Path(__file__).resolve()),'--root',str(args.root),'--run-id',args.run_id,'--case',case,'--arm',arm,'--qualification',str(args.qualification)]
            if args.fresh_conveyor_if_needed:command.append('--fresh-conveyor-if-needed')
            proc=subprocess.run(command)
            if proc.returncode:raise RuntimeError('Adjudication failed: '+case+'/'+arm)
            pending.remove((case,arm));progress=True
        if not pending:break
        if time.monotonic()>=deadline:
            print(json.dumps({'pending_missing_v1_or_runtime':sorted('/'.join(x) for x in pending)}),flush=True);return
        if not progress:time.sleep(min(10,max(.1,deadline-time.monotonic())))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/opt/astra-content-value-20260921'));p.add_argument('--run-id',default='pilot-v1');p.add_argument('--case',choices=CASES+['all'],default='all');p.add_argument('--arm',choices=['plain_astra','content_agents','both'],default='both');p.add_argument('--snapshot',type=Path);p.add_argument('--inventory',type=Path);p.add_argument('--source-root',type=Path);p.add_argument('--v1-output',type=Path);p.add_argument('--output',type=Path);p.add_argument('--qualification',type=Path,default=HERE/'qualification.json');p.add_argument('--qualify',action='store_true');p.add_argument('--wait-seconds',type=float,default=0);p.add_argument('--solver-python',type=Path);p.add_argument('--fresh-conveyor-if-needed',action='store_true');args=p.parse_args()
    if os.getpriority(os.PRIO_PROCESS,0)<15:os.nice(15-os.getpriority(os.PRIO_PROCESS,0))
    if args.qualify:
        assert args.snapshot,'Qualification needs an explicit verified frozen snapshot'
        qualify(args)
    elif args.case=='all' or args.arm=='both':watch_all(args)
    else:adjudicate(args)
