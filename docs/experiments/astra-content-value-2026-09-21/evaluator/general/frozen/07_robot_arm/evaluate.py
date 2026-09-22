"""Independent acceptance entrypoint. Run remotely with repo Python."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import traceback
import numpy as np
from common import SEEDS,check,dump,sha,verdict

HERE=Path(__file__).resolve().parent

def observe(result,config):
    checks=[];contract=config['contract'];c=contract['common'];seed=result['seed'];prefix=f'seed{seed}:'
    def chk(name,passed,evidence=None,insufficient=False):check(checks,prefix+name,passed,evidence,insufficient)
    if 'infrastructure_error' in result:
        chk('solver_run',False,result['infrastructure_error'],True);return checks
    chk('direct_run_completed',result['completed_steps']==result['expected_steps'],result['completed_steps'])
    chk('finite_solver_state',result['finite_states_and_contacts'])
    for field,limit in [('max_linear_speed_m_s',c['max_linear_speed_m_s']),('max_angular_speed_rad_s',c['max_angular_speed_rad_s']),('max_joint_closure_m',c['max_joint_closure_m']),('max_joint_axis_error_rad',c['max_joint_axis_error_rad']),('max_contact_penetration_m',c['max_penetration_m'])]:
        chk(field,result[field]<=limit,{'measured':result[field],'limit':limit})
    for name in ['native_mass','native_inertia']:
        values=[v for a in result[name].values() for v in a]
        # Inertia tensor may be a full symmetric matrix with zero off-diagonals.
        okay=bool(values) and np.isfinite(values).all()
        if name=='native_mass':okay &= all(v>0 for v in values)
        else:
            for a in result[name].values():
                matrix=np.asarray(a).reshape(3,3) if len(a)==9 else np.diag(a)
                okay &= bool(np.all(np.linalg.eigvalsh(matrix)>0))
        chk(name+'_finite_positive',okay)
    if result['completed_steps']<result['expected_steps']:return checks
    witness=result['gravity_witness'];w=witness['at_point_one_s'];expected=-9.81*w['t']
    vz=w['velocity'][2];dz=w['pose'][2]-witness['initial_pose'][2]
    chk('independent_freefall_witness',abs(vz-expected)<.03 and abs(dz+.5*9.81*w['t']**2)<.004,{'time':w['t'],'delta_z':dz,'vz':vz,'expected_vz':expected})
    trace=result['trace'];times=np.array([x['t'] for x in trace]);roles=config['joint_roles']
    def qs(role):return np.array([x['joints'][roles[role]]['q_unwrapped'] for x in trace])
    for path,joint in config['joints'].items():
        if joint['kind']=='fixed':continue
        lower,upper=joint.get('lower'),joint.get('upper');lo,hi=result['joint_extrema'][path]
        if lower is not None and upper is not None and lower<=upper:
            tolerance=c['angle_tolerance_rad'] if joint['kind']=='revolute' else c['position_tolerance_m']
            chk('joint_limits:'+path,lo>=lower-tolerance and hi<=upper+tolerance,{'observed':[lo,hi],'authored':[lower,upper],'tolerance':tolerance})
    observer=contract['observer']
    if observer=='door':
        q=qs('hinge');start=q[0];closed=q[(times>=9.5)&(times<10)]
        chk('door_started_closed',abs(start)<=contract['close_angle_rad'],float(start))
        chk('door_opened_60deg',np.max(q[(times>=1)&(times<7)])>=contract['open_angle_rad'],float(q.max()))
        chk('door_closed_5deg',len(closed)>0 and np.max(np.abs(closed))<=contract['close_angle_rad'],closed.tolist())
        hinge=config['joints'][roles['hinge']]
        chk('door_authored_finite_limits',hinge.get('lower') is not None and hinge.get('upper') is not None)
        if hinge.get('upper') is not None:
            chk('door_upper_limit_exercised',np.max(q[times>=10])>=hinge['upper']-.1,{'upper':hinge['upper'],'observed':float(np.max(q[times>=10]))},True)
    elif observer=='engine':
        crank,piston,beam=qs('crank'),qs('piston_slide'),qs('beam_pivot')
        cycle=float(crank.max()-crank[0]);chk('full_crank_cycle',cycle>=contract['minimum_crank_cycle_rad'],cycle)
        chk('piston_stroke',np.ptp(piston)>=contract['minimum_piston_stroke_m'],float(np.ptp(piston)))
        chk('beam_swing',np.ptp(beam)>=contract['minimum_beam_swing_rad'],float(np.ptp(beam)))
        cross=np.where(crank>=crank[0]+2*np.pi)[0]
        if len(cross):
            k=cross[0];f=(crank[0]+2*np.pi-crank[k-1])/(crank[k]-crank[k-1]);returned=piston[k-1]+f*(piston[k]-piston[k-1])
            chk('piston_cycle_repeat',abs(returned-piston[0])<=c['position_tolerance_m'],float(returned-piston[0]))
        else:chk('piston_cycle_repeat',False,'No complete cycle occurred')
    elif observer in ('arm','slider','axes'):
        for role in contract['required_joint_roles']:
            q=qs(role);kind=config['joints'][roles[role]]['kind'];minimum=contract.get('minimum_each_motion_rad',contract.get('minimum_travel_m',contract.get('minimum_each_travel_m',.02)))
            chk('required_joint_motion:'+role,np.ptp(q[(times>=1)&(times<7)])>=minimum,float(np.ptp(q[(times>=1)&(times<7)])))
            holding=[abs(row['joints'][roles[role]]['q_unwrapped']-row['joints'][roles[role]]['target']) for row in trace if 5.7<=row['t']<6.5]
            tolerance=c['angle_tolerance_rad'] if kind=='revolute' else c['position_tolerance_m']
            chk('target_holding_under_load:'+role,bool(holding) and max(holding)<=tolerance,{'max_error':max(holding) if holding else None,'tolerance':tolerance})
            returned=q[(times>=9.6)&(times<10)]
            chk('joint_return:'+role,bool(len(returned)) and np.max(np.abs(returned-q[0]))<=tolerance,returned.tolist())
        if observer=='arm':
            ee=config['body_roles']['end_effector'];points=np.array([row['poses'][ee][:3] for row in trace]);travel=float(np.max(np.linalg.norm(points-points[0],axis=1)))
            chk('end_effector_moved',travel>=contract['minimum_end_effector_motion_m'],travel)
        if observer=='slider':
            role=contract['required_joint_roles'][0];j=config['joints'][roles[role]];q=qs(role)
            chk('slider_authored_finite_limits',j.get('lower') is not None and j.get('upper') is not None)
            if j.get('upper') is not None:chk('slider_upper_limit_exercised',np.max(q[times>=10])>=j['upper']-.002,{'upper':j['upper'],'observed':float(np.max(q[times>=10]))},True)
    elif observer=='unsupported':
        chk('task_observer_available',False,contract['unsupported_reason'],True)
    else:chk('task_observer_available',False,'Unknown observer '+observer,True)
    # Fixed base means base explicitly declared nonmoving and externally anchored.
    base=config['body_roles'].get('base')
    if base:
        positions=np.array([row['poses'][base][:3] for row in trace]);travel=float(np.max(np.linalg.norm(positions-positions[0],axis=1)))
        chk('base_stayed_fixed',travel<=c['max_fixed_base_motion_m'],travel)
    return checks

def evaluate(args):
    from structural import inspect
    import jsonschema
    args.output.mkdir(parents=True,exist_ok=False)
    bindings=json.loads(args.bindings.read_text());inventory=json.loads(args.inventory.read_text());cases=json.loads((HERE/'cases.json').read_text())
    checks=[]
    frozen_path=HERE/'frozen_manifest.json'
    if frozen_path.exists():
        frozen=json.loads(frozen_path.read_text())
        check(checks,'reference_inventory_frozen',sha(args.inventory)==frozen['reference_inventory_sha256'])
        for filename,digest in frozen['code_sha256'].items():check(checks,'evaluator_snapshot:'+filename,sha(HERE/filename)==digest,insufficient=True)
    try:jsonschema.validate(bindings,json.loads((HERE/'bindings.schema.json').read_text()));valid=True
    except jsonschema.ValidationError as exc:valid=False;check(checks,'bindings_schema',False,str(exc))
    if not valid:
        dump(args.output/'acceptance.json',verdict(checks));return
    check(checks,'case_identity',bindings['case_id']==inventory['case_id']==args.case)
    if args.case=='04_gripper':cases['cases'][args.case]=json.loads((HERE/'case04_contract.json').read_text())
    contract=dict(cases['cases'][args.case]);contract['common']=cases['common'];contract['case_id']=args.case
    source_root=Path(args.source_root or inventory['source_root'])
    try:structural,config=inspect(args.usd,bindings,contract,inventory,source_root,args.inventory.parent,args.output);checks.extend(structural)
    except Exception as exc:
        check(checks,'structural_evaluator_completed',False,{'error':str(exc),'traceback':traceback.format_exc()},True);config=None
    if config and args.case=='04_gripper':
        from case04_evaluate import prepare
        try:
            if not prepare(config,bindings,inventory,checks,args.output):config=None
        except Exception as exc:
            check(checks,'case04_preparation_completed',False,{'error':str(exc),'traceback':traceback.format_exc()},True);config=None
    unsafe=not config or any(not x['passed'] and x['name'].startswith(('usd_stage_load','body_exists','rigid_body_binding','finite_rigid_transform','positive_finite_mass_inertia','joint_body_known','required_joint_roles','required_body_roles','meters_and_z_up')) for x in checks)
    if not unsafe and not args.structural_only:
        for seed in SEEDS:
            result_file=args.output/f'seed_{seed}.json';log_file=args.output/f'seed_{seed}.log'
            with log_file.open('w') as log:
                proc=subprocess.run([str(args.solver_python),str(HERE/('case04_solver.py' if args.case=='04_gripper' else 'solver.py')),'--config',str(args.output/'runtime_config.json'),'--seed',str(seed),'--output',str(result_file),'--device',args.device],stdout=log,stderr=subprocess.STDOUT,timeout=900)
            if not result_file.exists():
                check(checks,f'seed{seed}:solver_process',False,{'returncode':proc.returncode,'log':str(log_file)},True);continue
            result=json.loads(result_file.read_text())
            if args.case=='04_gripper':
                from case04_evaluate import observe as observe04,replay
                checks.extend(observe04(result,config))
                if seed==11 and 'trace' in result:replay(result,config,args.output/'replay_seed11.usda')
            else:checks.extend(observe(result,config))
    else:check(checks,'direct_task_runtime',False,'Structural-only mode or unsafe/missing runtime bindings; no functional acceptance.',True)
    if args.case=='06_engine' and not unsafe and not args.structural_only:
        baseline=dict(config);baseline['external_load_scale']=0.0;baseline_config=args.output/'unloaded_config.json';dump(baseline_config,baseline)
        result_file=args.output/'unloaded_seed11.json'
        with (args.output/'unloaded_seed11.log').open('w') as log:
            subprocess.run([str(args.solver_python),str(HERE/'solver.py'),'--config',str(baseline_config),'--seed','11','--output',str(result_file),'--device',args.device],stdout=log,stderr=subprocess.STDOUT,timeout=900)
        loaded_path=args.output/'seed_11.json'
        if result_file.exists() and loaded_path.exists():
            unloaded=json.loads(result_file.read_text());loaded=json.loads(loaded_path.read_text())
            if 'trace' in unloaded and 'trace' in loaded:
                path=config['joint_roles']['crank'];differences=[abs(a['joints'][path]['q_unwrapped']-b['joints'][path]['q_unwrapped']) for a,b in zip(loaded['trace'],unloaded['trace']) if 5.6<=a['t']<=6.5]
                check(checks,'engine_paired_load_response',bool(differences) and max(differences)>=1e-4,{'max_angle_change_rad':max(differences) if differences else None,'required_rad':1e-4,'input':'same seed, scene and controller; frozen inertia-normalized external shaft torque removed only in paired baseline'})
            else:check(checks,'engine_paired_load_response',False,'Paired solver run unavailable',True)
        else:check(checks,'engine_paired_load_response',False,'Paired solver output missing',True)
    report=verdict(checks)
    report.update({'case_id':args.case,'seeds':SEEDS,'input_sha256':sha(args.usd),'bindings_sha256':sha(args.bindings),'inventory_sha256':sha(args.inventory),'cases_sha256':sha(HERE/'cases.json'),
                   'evidence_policy':'Only evaluator-owned source inspection and fresh solver trajectories used. Inconclusive capability checks are not failures or false passes. A false-pass comparison additionally requires the author arm explicit accepted claim and at least one concrete failing check.'})
    dump(args.output/'acceptance.json',report)
    print(json.dumps({k:report[k] for k in ('case_id','accepted','status','concrete_failures','inconclusive_checks')},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--usd',type=Path,required=True);p.add_argument('--bindings',type=Path,required=True);p.add_argument('--inventory',type=Path,required=True);p.add_argument('--case',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--source-root');p.add_argument('--solver-python',type=Path,default=Path('/opt/astra-content-value-20260921/ovphysx-venv/bin/python'));p.add_argument('--device',default='cpu');p.add_argument('--structural-only',action='store_true');evaluate(p.parse_args())
